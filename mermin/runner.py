"""High-level runner for the Mermin inequality test on IQM hardware.

Workflow:
  1. Validate n (early — before allocating 2^n circuits).
  2. Resolve a backend (explicit / IQM cloud / AerSimulator fallback).
  3. Resolve the device topology, *always data-driven*:
       - live IQM backend     → `topology_from_iqm_backend(backend)`
       - calibration JSON path → `topology_from_calibration_json(path)`
       - simulator             → tiny linear-chain placeholder
  4. Annotate the topology with calibration data (live or from JSON).
  5. Choose a chain of n qubits via:
       - IQM Qubit Selector    (canonical, calibration-aware)
       - user-supplied chain   (validated against the live topology)
       - local DFS selector    (fidelity-product fallback)
  6. Build one circuit per nonzero term of M_n.
  7. Transpile + run + aggregate.

Result includes a per-term breakdown so users can plug in their own error
mitigation if needed.
"""

from __future__ import annotations

from typing import Optional, Sequence, List, Dict, Tuple, Any

import warnings

from qiskit import transpile

from polynomial import MerminProblem
from topology import (
    Topology,
    topology_from_iqm_backend,
    topology_from_calibration_json,
    parse_calibration_json,
    annotate_topology_with_calibration,
    fetch_calibration_from_backend,
    select_chain,
    find_best_chain_path,
    validate_user_chain,
    ChainSelection,
)
from circuits import build_all_circuits
from analysis import (
    expectation_from_counts,
    TermResult,
    MerminResult,
)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_backend(
    backend=None,
    iqm_server_url: Optional[str] = None,
    iqm_token: Optional[str] = None,
    iqm_device: Optional[str] = None,
) -> Tuple[Any, str, bool]:
    """Return (backend, name, is_simulator)."""
    if backend is not None:
        name = getattr(backend, "name", "user-supplied")
        if callable(name):
            name = name()
        is_sim = "aer" in str(name).lower() or "simul" in str(name).lower()
        return backend, str(name), is_sim

    import os as _os
    # Token can come from the explicit arg OR from IQM_TOKEN in the environment.
    # Pass token=None to IQMProvider when relying on the env var so IQMClient
    # doesn't see the token from two sources at once (raises ClientConfigurationError).
    _env_token = _os.environ.get("IQM_TOKEN")
    _pass_token = iqm_token  # None → IQMClient reads IQM_TOKEN from env itself
    if iqm_server_url and (iqm_token or _env_token):
        try:
            from iqm.qiskit_iqm import IQMProvider  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "iqm-client[qiskit] not available. Install it (Python 3.11 "
                "recommended) with:\n"
                "    pip install 'iqm-client[qiskit]>=32.1.1,<33.0'"
            ) from e
        try:
            if iqm_device:
                provider = IQMProvider(
                    iqm_server_url, token=_pass_token, quantum_computer=iqm_device
                )
            else:
                provider = IQMProvider(iqm_server_url, token=_pass_token)
        except TypeError:
            provider = IQMProvider(iqm_server_url, token=_pass_token)
        try:
            be = provider.get_backend(iqm_device) if iqm_device else provider.get_backend()
        except TypeError:
            be = provider.get_backend()
        nm = getattr(be, "name", str(iqm_device or "iqm"))
        if callable(nm):
            nm = nm()
        return be, str(nm), False

    from qiskit_aer import AerSimulator
    return AerSimulator(), "AerSimulator", True


# ---------------------------------------------------------------------------
# Topology resolution (always data-driven)
# ---------------------------------------------------------------------------

def _resolve_topology(
    backend,
    is_simulator: bool,
    n: int,
    *,
    explicit: Optional[Topology] = None,
    calibration_json: Optional[str] = None,
    fetch_live_calibration: bool = True,
) -> Topology:
    """Pick a Topology, *never* using hardcoded coupling maps.

    Preference:
      1. Explicit `topology=` argument
      2. Live `backend.coupling_map` (real IQM hardware)
      3. Topology derived from `calibration_json`
      4. Tiny linear-chain placeholder for the simulator
    """
    if explicit is not None:
        top = explicit
    elif not is_simulator and hasattr(backend, "coupling_map"):
        try:
            top = topology_from_iqm_backend(backend)
        except Exception as e:
            warnings.warn(f"could not derive topology from backend ({e})")
            if calibration_json:
                top = topology_from_calibration_json(calibration_json)
            else:
                raise
    elif calibration_json:
        top = topology_from_calibration_json(calibration_json)
    else:
        # Simulator + no calibration: simple line of length n.
        edges = [(i, i + 1) for i in range(n - 1)]
        names = {i: f"QB{i+1}" for i in range(n)}
        top = Topology.from_edges(
            name="simulator-line",
            edges=edges,
            nodes=range(n),
            qubit_names=names,
        )

    # Hydrate calibration numbers onto the topology
    if calibration_json:
        try:
            cal = parse_calibration_json(calibration_json)
            annotate_topology_with_calibration(top, cal)
        except Exception as e:
            warnings.warn(f"failed to load calibration {calibration_json}: {e}")
    elif not is_simulator and fetch_live_calibration:
        try:
            cal = fetch_calibration_from_backend(backend)
            annotate_topology_with_calibration(top, cal)
        except Exception as e:
            warnings.warn(
                f"could not fetch live calibration from backend ({e}); "
                f"will use uniform fidelity defaults"
            )

    return top


# ---------------------------------------------------------------------------
# Hardware-feasibility checks
# ---------------------------------------------------------------------------

def _validate_feasibility(
    n: int, backend, is_simulator: bool, top: Topology,
) -> List[str]:
    notes: List[str] = []
    nq = getattr(backend, "num_qubits", None)
    if nq is None and hasattr(backend, "configuration"):
        try:
            nq = backend.configuration().n_qubits
        except Exception:
            nq = None
    if nq is not None and n > nq:
        raise ValueError(f"backend has {nq} qubits but n={n} were requested")
    if n > len(top.nodes):
        raise ValueError(
            f"topology {top.name!r} has {len(top.nodes)} qubits but n={n} requested"
        )
    if n > 16 and is_simulator:
        notes.append(f"n={n} on the statevector simulator may be slow")
    return notes


# ---------------------------------------------------------------------------
# Chain selection dispatcher
# ---------------------------------------------------------------------------

def _select_chain_dispatch(
    backend,
    is_sim: bool,
    top: Topology,
    n: int,
    user_chain: Optional[Sequence],
    strategy: str,
    iqm_selector_kwargs: Dict[str, Any],
) -> ChainSelection:
    """Apply chain-selection strategy with proper fallbacks.

    Strategies:
      'auto'          : local DFS over `top`
      'user'          : require valid user_chain
      'user_or_auto'  : user_chain → local DFS
      'iqm'           : require IQM Qubit Selector (no fallback)
      'iqm_or_auto'   : IQM selector → local DFS  (recommended for HW)
      'user_or_iqm'   : user_chain → IQM selector → local DFS
    """
    # IQM selector needs a real backend
    if is_sim and strategy in ("iqm", "iqm_or_auto", "user_or_iqm"):
        strategy = "user_or_auto" if strategy == "user_or_iqm" else "auto"

    if strategy in ("auto", "user", "user_or_auto"):
        return select_chain(top, n, user_chain=user_chain, strategy=strategy)

    from qubit_selector_adapter import select_chain_iqm

    def _try_user() -> Optional[ChainSelection]:
        if user_chain is None:
            return None
        return validate_user_chain(top, n, user_chain)

    def _try_iqm() -> Optional[ChainSelection]:
        try:
            return select_chain_iqm(backend, n, **iqm_selector_kwargs)
        except Exception as e:
            warnings.warn(f"IQM Qubit Selector failed: {e}")
            return None

    if strategy == "iqm":
        sel = _try_iqm()
        if sel is None:
            raise RuntimeError("IQM Qubit Selector failed (no fallback allowed)")
        return sel
    if strategy == "iqm_or_auto":
        sel = _try_iqm()
        if sel is not None:
            return sel
        return find_best_chain_path(top, n)
    if strategy == "user_or_iqm":
        sel = _try_user()
        if sel is not None:
            return sel
        sel = _try_iqm()
        if sel is not None:
            return sel
        return find_best_chain_path(top, n)
    raise ValueError(f"unknown chain_strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_mermin_test(
    n: int,
    *,
    # backend selection
    backend=None,
    iqm_server_url: Optional[str] = None,
    iqm_token: Optional[str] = None,
    iqm_device: Optional[str] = None,
    # qubit-chain selection
    qubit_chain: Optional[Sequence] = None,
    chain_strategy: str = "iqm_or_auto",
    iqm_selector_kwargs: Optional[Dict[str, Any]] = None,
    # topology / calibration
    topology: Optional[Topology] = None,
    calibration_json: Optional[str] = None,
    fetch_live_calibration: bool = True,
    # measurement
    shots: int = 4096,
    optimize_angles: bool = False,
    # transpilation
    optimization_level: int = 3,
    transpile_kwargs: Optional[Dict[str, Any]] = None,
    # behaviour
    verbose: bool = True,
) -> MerminResult:
    """Run an n-qubit Mermin inequality test."""
    # ---- 1. Early validation -------------------------------------------
    if n < 2:
        raise ValueError(f"n must be >= 2 (got {n})")
    if n > 20:
        raise ValueError(
            f"n={n} impractical: M_n has up to 2^{n} circuits; pick n<=20"
        )

    # ---- 2. Backend ----------------------------------------------------
    backend, backend_name, is_sim = _resolve_backend(
        backend=backend,
        iqm_server_url=iqm_server_url,
        iqm_token=iqm_token,
        iqm_device=iqm_device,
    )
    if verbose:
        print(f"[backend] {backend_name}  (simulator={is_sim})")

    # ---- 3. Topology (data-driven) -------------------------------------
    top = _resolve_topology(
        backend, is_sim, n,
        explicit=topology,
        calibration_json=calibration_json,
        fetch_live_calibration=fetch_live_calibration,
    )
    if verbose:
        print(f"[topology] {top}")

    # ---- 4. Feasibility -------------------------------------------------
    notes = _validate_feasibility(n, backend, is_sim, top)

    # ---- 5. Polynomial + angles ----------------------------------------
    problem = MerminProblem.make(n, optimize=optimize_angles)
    if verbose:
        print(problem.summary())
    for note in notes:
        if verbose:
            print(f"[note] {note}")

    # ---- 6. Chain selection --------------------------------------------
    sel = _select_chain_dispatch(
        backend, is_sim, top, n,
        user_chain=qubit_chain,
        strategy=chain_strategy,
        iqm_selector_kwargs=iqm_selector_kwargs or {},
    )
    if verbose:
        names = sel.chain_names(top)
        print(f"[chain] indices {sel.chain}  names {names}")
        print(f"        score={sel.score:.4f}  reason={sel.reason}")

    # ---- 7. Build circuits ---------------------------------------------
    compiled = build_all_circuits(
        n=n,
        poly=problem.poly,
        alpha=problem.alpha,
        beta=problem.beta,
        physical_qubits=sel.chain,
    )
    # Force a clean virtual->physical map (chain[i] is the qiskit physical
    # index for virtual i; topology indices are 0-based now)
    compiled.initial_layout = {v: int(p) for v, p in enumerate(sel.chain)}
    if verbose:
        print(f"[circuits] {len(compiled.circuits)} term-circuits built")

    # ---- 8. Transpile ---------------------------------------------------
    tkw = dict(transpile_kwargs or {})
    tkw.setdefault("optimization_level", optimization_level)
    if not is_sim and hasattr(backend, "coupling_map"):
        tkw.setdefault("initial_layout", list(compiled.initial_layout.values()))
    try:
        transpiled = transpile(compiled.circuits, backend=backend, **tkw)
    except Exception as e:
        warnings.warn(
            f"transpile with initial_layout failed ({e!r}); retrying without it"
        )
        tkw.pop("initial_layout", None)
        transpiled = transpile(compiled.circuits, backend=backend, **tkw)

    # ---- 9. Run --------------------------------------------------------
    job = backend.run(transpiled, shots=shots)
    if verbose:
        print(f"[job] submitted; waiting for results...")
    qresult = job.result()

    # ---- 10. Aggregate (indexed lookup; IQM strips circuit names) ------
    try:
        all_counts = qresult.get_counts()
    except Exception:
        all_counts = None
    if isinstance(all_counts, dict):
        all_counts = [all_counts]

    term_results: List[TermResult] = []
    for idx, (pat, coef) in enumerate(
        zip(compiled.patterns, compiled.coefficients)
    ):
        if all_counts is not None and idx < len(all_counts):
            counts = all_counts[idx]
        else:
            counts = qresult.get_counts(idx)
            if isinstance(counts, list):
                counts = counts[0]
        clean_counts = {k.replace(" ", ""): v for k, v in counts.items()}
        mean, sem = expectation_from_counts(clean_counts)
        term_results.append(
            TermResult(
                pattern=pat,
                coefficient=coef,
                expectation=mean,
                sem=sem,
                counts=clean_counts,
            )
        )

    chain_names = sel.chain_names(top)
    result = MerminResult(
        problem=problem,
        physical_qubits=sel.chain,
        backend_name=backend_name,
        shots=shots,
        term_results=term_results,
        chain_reason=f"{sel.reason}  |  qubits: {chain_names}",
        notes=notes,
    )
    if verbose:
        print(result.summary())
    return result
