"""Adapter for IQM's official `iqm-qubit-selector` library.

Reference: https://docs.iqm.tech/iqm-qubit-selector/

This module is a thin wrapper around `CostEvaluator` so that we can use
IQM's own calibration-driven layout chooser to pick the best path for our
GHZ-prep circuit. Falls back to the local DFS selector when the library
isn't installed.

Usage:

    from mermin_iqm.qubit_selector_adapter import select_chain_iqm
    sel = select_chain_iqm(backend, n=4)        # uses linear-CX GHZ
    print(sel.chain, sel.chain_names(topology))
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from qiskit import QuantumCircuit

from topology import (
    Topology,
    ChainSelection,
    find_best_chain_path,
)
from circuits import ghz_state_circuit


def _ghz_template_for_qubit_selector(n: int) -> QuantumCircuit:
    """Build a *measurement-free* linear-CX GHZ circuit, the exact connectivity
    pattern Mermin uses. Stripping measurements lets the IQM selector focus
    purely on the 2q-gate connectivity requirements."""
    qc = ghz_state_circuit(n)  # H on q0 + CX chain q0->q1->q2->...
    return qc


def select_chain_iqm(
    backend,
    n: int,
    *,
    cost_function: str = "gate_cost_cz",   # 'gate_cost_cz' or 'gate_cost_clifford'
    readout_mode: str = "fidelity",        # 'none', 'fidelity', 'qndness'
    num_trials: int = 2000,
    num_layouts: int = 10,
    remove_qubit_names: Optional[Sequence[str]] = None,
) -> ChainSelection:
    """Use the official IQM Qubit Selector to pick the best path for a
    linear-CX GHZ circuit on `backend`.

    Parameters
    ----------
    backend
        A live IQM backend (e.g. `provider.get_backend("garnet")`).
    n
        Number of qubits in the chain.
    cost_function
        Which 2q-fidelity to optimize for. 'gate_cost_cz' (default) ranks
        layouts by Π (CZ fidelity); 'gate_cost_clifford' uses the
        Clifford-averaged fidelity.
    readout_mode
        Whether to fold readout fidelity into the cost. 'fidelity' (default)
        multiplies by readout fidelity per-qubit; 'qndness' uses QND-ness;
        'none' ignores readout (matches the IQM docs default).
    num_trials
        Layouts to enumerate before scoring (IQM's `num_trials` argument).
    num_layouts
        How many ranked layouts to fetch back (we only use the top one).
    remove_qubit_names
        IQM-style names of qubits to exclude from consideration (e.g.
        ['QB1', 'QB2']). Useful when calibration shows a known-bad qubit.

    Returns
    -------
    ChainSelection
        With `.chain` as 0-indexed qiskit qubit indices in chain order and
        `.score = 1 - cost` (so larger is better, matching the local
        selector's convention).

    Raises
    ------
    RuntimeError
        If `iqm-qubit-selector` isn't installed.
    """
    try:
        from iqm.qubit_selector.qubit_selector import (  # type: ignore
            CostEvaluator, CostFunction, ReadoutMode,
        )
    except Exception as e:
        raise RuntimeError(
            "iqm-qubit-selector not available. Install it with:\n"
            "    pip install iqm-qubit-selector\n"
            "or use chain_strategy='auto' for the local selector."
        ) from e

    # Translate string args to the library's enums
    cf_map = {
        "gate_cost_cz": CostFunction.GATE_COST_CZ,
        "gate_cost_clifford": CostFunction.GATE_COST_CLIFFORD,
    }
    rm_map = {
        "none": ReadoutMode.NONE,
        "fidelity": ReadoutMode.FIDELITY,
        "qndness": ReadoutMode.QNDNESS,
    }
    cf = cf_map.get(cost_function.lower())
    if cf is None:
        raise ValueError(f"unknown cost_function {cost_function!r}; "
                         f"use one of {list(cf_map)}")
    rm = rm_map.get(readout_mode.lower())
    if rm is None:
        raise ValueError(f"unknown readout_mode {readout_mode!r}; "
                         f"use one of {list(rm_map)}")

    # Convert qubit-name removals to qiskit indices
    remove_qubits: Optional[List[int]] = None
    if remove_qubit_names:
        remove_qubits = []
        for name in remove_qubit_names:
            try:
                remove_qubits.append(backend.qubit_name_to_index(name))
            except Exception:
                # silently skip unknown qubits
                pass

    qc = _ghz_template_for_qubit_selector(n)
    evaluator = CostEvaluator(
        backend=backend,
        quantum_circuit=qc,
        cost_function=cf,
        readoutmode=rm,
        remove_qubits=remove_qubits,
        num_trials=num_trials,
    )
    layouts, costs = evaluator.get_top_layouts(num_layouts=num_layouts)
    if not layouts:
        raise RuntimeError(
            f"IQM Qubit Selector returned no layouts for n={n} GHZ on "
            f"{getattr(backend, 'name', None)}; the device may be too "
            f"small or all qubits were removed."
        )
    best = layouts[0]
    cost = float(costs[0])

    # `cost` is a fraction in [0, 1] where 0 == perfect; flip to a 'score'
    score = max(0.0, 1.0 - cost)
    return ChainSelection(
        chain=[int(q) for q in best],
        score=score,
        reason=(f"IQM Qubit Selector: cost={cost*100:.2f}% "
                f"(cf={cost_function}, readout={readout_mode}, "
                f"trials={num_trials})"),
    )