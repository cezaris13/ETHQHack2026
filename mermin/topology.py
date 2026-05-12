"""Device topology and calibration handling for IQM Crystal-class QPUs.

**No hardcoded coupling maps.** The topology is always derived from one of:

  1. A live `IQMBackendBase` (uses `backend.coupling_map`).
  2. The official IQM calibration JSON downloaded from Resonance
     (the CZ pairs in the calibration *define* the coupling graph).
  3. Anything compatible with qiskit's `CouplingMap.get_edges()`.

Calibration data is parsed in the same format the official
`iqm-qubit-selector` library uses (`CalibrationDataManager`):

    {
        'CZ':       {'[QB1, QB2]': 0.991, '[QB2, QB1]': 0.991, ...},
        'CLIFFORD': {'[QB1, QB2]': 0.985, ...},
        '1Q':       {'QB1': 0.9991, ...},
        'readout':  {'QB1': 0.97, ...},
        't1':       {'QB1': 50.2, ...},   # in microseconds
        't2':       {'QB1': 25.0, ...},
    }

But we also accept the *raw observation list* shape that Resonance's
"Download calibration data" button currently emits, where each entry
has a `dut_field` string like `"QB1.QB2.cz.fidelity"` and a `value`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple, Dict, Optional, Any

import json
import re


Edge = Tuple[int, int]


# ---------------------------------------------------------------------------
# Topology data structure
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """Undirected coupling graph for a quantum device.

    Nodes are integer qubit IDs (0-indexed by convention to match qiskit).
    Edges are stored as a dict adjacency list. Per-qubit and per-edge
    fidelity numbers (and T1/T2) come from the calibration loader.
    """
    name: str
    nodes: List[int]
    adjacency: Dict[int, set] = field(default_factory=dict)
    fidelity_1q: Dict[int, float] = field(default_factory=dict)
    fidelity_2q: Dict[Edge, float] = field(default_factory=dict)
    fidelity_readout: Dict[int, float] = field(default_factory=dict)
    t1: Dict[int, float] = field(default_factory=dict)
    t2: Dict[int, float] = field(default_factory=dict)
    # Map between canonical IQM qubit name ("QB7") and integer index
    qubit_name_to_index: Dict[str, int] = field(default_factory=dict)
    qubit_index_to_name: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        name: str,
        edges: Iterable[Edge],
        nodes: Optional[Iterable[int]] = None,
        qubit_names: Optional[Dict[int, str]] = None,
    ) -> "Topology":
        edges = [tuple(sorted(map(int, e))) for e in edges]
        nodes_set = set()
        for a, b in edges:
            nodes_set.add(a); nodes_set.add(b)
        if nodes is not None:
            nodes_set.update(int(q) for q in nodes)
        nodes_list = sorted(nodes_set)
        adj: Dict[int, set] = {q: set() for q in nodes_list}
        for a, b in edges:
            if a == b:
                continue
            adj[a].add(b); adj[b].add(a)
        idx_to_name = dict(qubit_names) if qubit_names else {}
        name_to_idx = {v: k for k, v in idx_to_name.items()}
        return cls(
            name=name,
            nodes=nodes_list,
            adjacency=adj,
            qubit_index_to_name=idx_to_name,
            qubit_name_to_index=name_to_idx,
        )

    @property
    def edges(self) -> List[Edge]:
        seen, out = set(), []
        for a, neighbours in self.adjacency.items():
            for b in neighbours:
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    def has_edge(self, a: int, b: int) -> bool:
        return b in self.adjacency.get(a, set())

    def neighbours(self, a: int) -> set:
        return set(self.adjacency.get(a, set()))

    def is_connected_chain(self, chain: Sequence[int]) -> bool:
        if len(chain) != len(set(chain)):
            return False
        for q in chain:
            if q not in self.adjacency:
                return False
        for a, b in zip(chain, chain[1:]):
            if not self.has_edge(a, b):
                return False
        return True

    def chain_score(self, chain: Sequence[int]) -> float:
        """Product of 1q × 2q (× readout) fidelities along chain.

        Missing values default to 0.99 (1q/CZ) and 0.97 (readout) so chains
        can still be ranked when coverage is incomplete.
        """
        s = 1.0
        for q in chain:
            s *= self.fidelity_1q.get(q, 0.99)
            if self.fidelity_readout:
                s *= self.fidelity_readout.get(q, 0.97)
        for a, b in zip(chain, chain[1:]):
            e = (a, b) if a <= b else (b, a)
            s *= self.fidelity_2q.get(e, 0.99)
        return s

    def name_of(self, q: int) -> str:
        return self.qubit_index_to_name.get(q, f"QB{q+1}")

    def __repr__(self) -> str:
        cal = []
        if self.fidelity_2q: cal.append(f"{len(self.fidelity_2q)} CZ")
        if self.fidelity_1q: cal.append(f"{len(self.fidelity_1q)} 1Q")
        if self.t1: cal.append(f"{len(self.t1)} T1")
        cal_str = " calibrated: " + ", ".join(cal) if cal else " no calibration"
        return (f"Topology({self.name!r}, {len(self.nodes)} qubits, "
                f"{len(self.edges)} edges,{cal_str})")


# ---------------------------------------------------------------------------
# Topology construction from a live IQM backend
# ---------------------------------------------------------------------------

def topology_from_iqm_backend(backend, name: Optional[str] = None) -> Topology:
    """Build a Topology from a live IQM/qiskit backend.

    Reads `backend.coupling_map` and, when present,
    `backend.index_to_qubit_name` so we can show 'QB7' labels in reports.
    """
    if name is None:
        n = getattr(backend, "name", None)
        if callable(n):
            n = n()
        name = str(n) if n is not None else "iqm-backend"

    cmap = backend.coupling_map
    if cmap is None:
        raise ValueError(
            f"backend {name!r} does not expose a coupling_map; cannot derive topology"
        )
    if hasattr(cmap, "get_edges"):
        raw_edges = list(cmap.get_edges())
    else:
        raw_edges = list(cmap)
    undirected = {tuple(sorted(map(int, e))) for e in raw_edges}

    nq = backend.num_qubits

    # qubit-name mapping (IQM backends expose index_to_qubit_name)
    idx_to_name: Dict[int, str] = {}
    for i in range(nq):
        if hasattr(backend, "index_to_qubit_name"):
            try:
                idx_to_name[i] = backend.index_to_qubit_name(i)
            except Exception:
                pass
    if not idx_to_name:
        idx_to_name = {i: f"QB{i+1}" for i in range(nq)}

    return Topology.from_edges(
        name=name,
        edges=list(undirected),
        nodes=range(nq),
        qubit_names=idx_to_name,
    )


# ---------------------------------------------------------------------------
# Calibration JSON parsing (IQM Resonance format)
# ---------------------------------------------------------------------------

_QB_RE = re.compile(r"QB(\d+)")


def _parse_qubit_label(label: Any) -> Optional[str]:
    """Coerce many possible qubit labels to canonical 'QBn' form."""
    if isinstance(label, int):
        return f"QB{label}"
    if isinstance(label, str):
        m = _QB_RE.search(label)
        if m:
            return f"QB{m.group(1)}"
        # bare integer string?
        s = label.strip().strip("[]")
        if s.isdigit():
            return f"QB{s}"
    return None


def _parse_pair_label(label: Any) -> Optional[Tuple[str, str]]:
    """Extract a (qubit_a, qubit_b) tuple of canonical qubit names.

    Handles '[QB1, QB2]', 'QB1-QB2', 'QB1_QB2', 'QB1.QB2.cz.fidelity', etc.
    """
    if isinstance(label, (list, tuple)) and len(label) == 2:
        a = _parse_qubit_label(label[0])
        b = _parse_qubit_label(label[1])
    elif isinstance(label, str):
        matches = _QB_RE.findall(label)
        if len(matches) < 2:
            return None
        a, b = f"QB{matches[0]}", f"QB{matches[1]}"
    else:
        return None
    if a is None or b is None:
        return None
    return (a, b)


def _name_to_index_factory(qubit_names: Iterable[str]) -> Dict[str, int]:
    """Map ['QB1','QB2','QB3'] to {'QB1':0, 'QB2':1, ...} (sorted by number)."""
    sorted_names = sorted(
        set(qubit_names),
        key=lambda s: int(_QB_RE.search(s).group(1)) if _QB_RE.search(s) else 0,
    )
    return {name: i for i, name in enumerate(sorted_names)}


def parse_calibration_json(path_or_dict) -> Dict[str, Dict[str, float]]:
    """Read a calibration dump and return a dict keyed by metric name.

    Output shape (matches `iqm-qubit-selector`):

        {
          'CZ':       {'[QB1, QB2]': 0.99, ...},   # both orderings stored
          'CLIFFORD': {'[QB1, QB2]': 0.985, ...},
          '1Q':       {'QB1': 0.999, ...},
          'readout':  {'QB1': 0.97, ...},
          't1':       {'QB1': 50.2, ...},          # microseconds
          't2':       {'QB1': 25.0, ...},
        }

    Accepts:
      * already-parsed dict in the above shape
      * Resonance "Download calibration data" JSON (list of observations)
      * legacy/loose forms with keys like 'cz_gate_fidelity', 't1', etc.
    """
    if isinstance(path_or_dict, (str, bytes)):
        with open(path_or_dict, "r") as f:
            blob = json.load(f)
    else:
        blob = path_or_dict

    cz: Dict[str, float] = {}
    clifford: Dict[str, float] = {}
    sqg: Dict[str, float] = {}
    readout: Dict[str, float] = {}
    t1: Dict[str, float] = {}
    t2: Dict[str, float] = {}

    def _store_pair(target: Dict[str, float], pair: Tuple[str, str], val: float) -> None:
        a, b = pair
        target[f"[{a}, {b}]"] = float(val)
        target[f"[{b}, {a}]"] = float(val)

    # ------- shape A: already-parsed dict ------
    if isinstance(blob, dict) and any(k in blob for k in ("CZ", "1Q", "t1")):
        return {
            "CZ": dict(blob.get("CZ", {})),
            "CLIFFORD": dict(blob.get("CLIFFORD", {})),
            "1Q": dict(blob.get("1Q", {})),
            "readout": dict(blob.get("readout", {})),
            "readout_qndness": dict(blob.get("readout_qndness", {})),
            "t1": dict(blob.get("t1", {})),
            "t2": dict(blob.get("t2", {})),
        }

    # ------- shape B: list of observation records ------
    observations: List[dict] = []
    if isinstance(blob, list):
        observations = blob
    elif isinstance(blob, dict):
        for key in ("observations", "metrics", "calibration_metrics"):
            if isinstance(blob.get(key), list):
                observations = blob[key]
                break

    if observations:
        for rec in observations:
            if not isinstance(rec, dict):
                continue
            dut = rec.get("dut_field") or rec.get("metric") or rec.get("name", "")
            val = rec.get("value", rec.get("v"))
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            qb_matches = _QB_RE.findall(str(dut))
            dut_l = str(dut).lower()
            if "cz" in dut_l and "fidelity" in dut_l and len(qb_matches) >= 2:
                _store_pair(cz, (f"QB{qb_matches[0]}", f"QB{qb_matches[1]}"), val)
            elif "clifford" in dut_l and len(qb_matches) >= 2:
                _store_pair(clifford, (f"QB{qb_matches[0]}", f"QB{qb_matches[1]}"), val)
            elif "readout" in dut_l and "fidelity" in dut_l and qb_matches:
                readout[f"QB{qb_matches[0]}"] = val
            elif ("prx" in dut_l or "1q" in dut_l or "single" in dut_l) and "fidelity" in dut_l and qb_matches:
                sqg[f"QB{qb_matches[0]}"] = val
            elif "t1" in dut_l and qb_matches:
                t1[f"QB{qb_matches[0]}"] = val * 1e6 if val < 1 else val
            elif "t2" in dut_l and qb_matches:
                t2[f"QB{qb_matches[0]}"] = val * 1e6 if val < 1 else val
        return {
            "CZ": cz, "CLIFFORD": clifford, "1Q": sqg,
            "readout": readout, "readout_qndness": {},
            "t1": t1, "t2": t2,
        }

    # ------- shape C: legacy "fields with friendly names" ------
    if isinstance(blob, dict):
        for k in ("prx_gate_fidelity", "single_qubit_fidelity", "1Q", "1q_fidelity"):
            if isinstance(blob.get(k), dict):
                for q, v in blob[k].items():
                    qn = _parse_qubit_label(q)
                    if qn is not None:
                        sqg[qn] = float(v)
                break
        for k in ("cz_gate_fidelity", "two_qubit_fidelity", "CZ"):
            if isinstance(blob.get(k), dict):
                for pair, v in blob[k].items():
                    p = _parse_pair_label(pair)
                    if p is not None:
                        _store_pair(cz, p, float(v))
                break
        for k in ("cliffords_averaged_gate_fidelity", "clifford_fidelity", "CLIFFORD"):
            if isinstance(blob.get(k), dict):
                for pair, v in blob[k].items():
                    p = _parse_pair_label(pair)
                    if p is not None:
                        _store_pair(clifford, p, float(v))
                break
        for k in ("single_qubit_readout_fidelity", "readout_fidelity", "readout"):
            if isinstance(blob.get(k), dict):
                for q, v in blob[k].items():
                    qn = _parse_qubit_label(q)
                    if qn is not None:
                        readout[qn] = float(v)
                break
        for k in ("t1", "T1", "t1_time"):
            if isinstance(blob.get(k), dict):
                for q, v in blob[k].items():
                    qn = _parse_qubit_label(q)
                    if qn is not None:
                        v = float(v); t1[qn] = v * 1e6 if v < 1 else v
                break
        for k in ("t2_echo", "t2_ramsey", "t2", "T2", "t2_time"):
            if isinstance(blob.get(k), dict):
                for q, v in blob[k].items():
                    qn = _parse_qubit_label(q)
                    if qn is not None:
                        v = float(v); t2[qn] = v * 1e6 if v < 1 else v
                break

    return {
        "CZ": cz, "CLIFFORD": clifford, "1Q": sqg,
        "readout": readout, "readout_qndness": {},
        "t1": t1, "t2": t2,
    }


def load_calibration_json(path: str) -> Dict[str, Dict[str, float]]:
    """Backwards-compatible alias for `parse_calibration_json(path)`."""
    return parse_calibration_json(path)


# ---------------------------------------------------------------------------
# Topology directly from a calibration JSON (no backend needed)
# ---------------------------------------------------------------------------

def topology_from_calibration(
    calibration: Dict[str, Dict[str, float]],
    name: str = "from-calibration",
) -> Topology:
    """Build a Topology purely from calibration data.

    The CZ-fidelity pairs *define* the coupling graph: an edge exists iff
    the calibration reports a CZ fidelity for it. Single-qubit qubits are
    added as isolated nodes if needed.
    """
    cz = calibration.get("CZ", {})
    sqg = calibration.get("1Q", {})

    pair_set: set = set()
    qubit_names: set = set()
    for key in cz.keys():
        p = _parse_pair_label(key)
        if p is None:
            continue
        a, b = p
        qubit_names.add(a); qubit_names.add(b)
        pair_set.add((a, b) if a <= b else (b, a))

    for q in sqg.keys():
        qn = _parse_qubit_label(q)
        if qn is not None:
            qubit_names.add(qn)

    name_to_idx = _name_to_index_factory(qubit_names)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    edges = [
        (name_to_idx[a], name_to_idx[b]) for a, b in pair_set
        if a in name_to_idx and b in name_to_idx
    ]

    top = Topology.from_edges(
        name=name,
        edges=edges,
        nodes=name_to_idx.values(),
        qubit_names=idx_to_name,
    )
    annotate_topology_with_calibration(top, calibration)
    return top


def topology_from_calibration_json(path: str, name: Optional[str] = None) -> Topology:
    """Convenience: parse the JSON at `path` and build a topology from it."""
    cal = parse_calibration_json(path)
    return topology_from_calibration(cal, name=name or f"calibration[{path}]")


# ---------------------------------------------------------------------------
# Annotate an existing topology with parsed calibration data
# ---------------------------------------------------------------------------

def annotate_topology_with_calibration(
    top: Topology,
    calibration: Dict[str, Dict[str, float]],
) -> Topology:
    """Copy CZ/1Q/T1/T2/readout from a parsed calibration dict onto a topology.

    Mutates `top` in place and returns it. Calibration entries that don't
    match a topology node/edge are skipped silently.
    """
    name_to_idx = top.qubit_name_to_index or _name_to_index_factory(
        [top.name_of(q) for q in top.nodes]
    )
    if not top.qubit_name_to_index:
        top.qubit_name_to_index = dict(name_to_idx)
        top.qubit_index_to_name = {v: k for k, v in name_to_idx.items()}

    for key, val in calibration.get("CZ", {}).items():
        p = _parse_pair_label(key)
        if p is None:
            continue
        a, b = p
        ia = name_to_idx.get(a); ib = name_to_idx.get(b)
        if ia is None or ib is None:
            continue
        if not top.has_edge(ia, ib):
            continue
        e = (ia, ib) if ia <= ib else (ib, ia)
        top.fidelity_2q[e] = float(val)

    for key, val in calibration.get("1Q", {}).items():
        qn = _parse_qubit_label(key)
        if qn is None:
            continue
        i = name_to_idx.get(qn)
        if i is None:
            continue
        top.fidelity_1q[i] = float(val)

    for key, val in calibration.get("readout", {}).items():
        qn = _parse_qubit_label(key)
        if qn is None:
            continue
        i = name_to_idx.get(qn)
        if i is None:
            continue
        top.fidelity_readout[i] = float(val)

    for src_key, dest in (("t1", top.t1), ("t2", top.t2)):
        for key, val in calibration.get(src_key, {}).items():
            qn = _parse_qubit_label(key)
            if qn is None:
                continue
            i = name_to_idx.get(qn)
            if i is None:
                continue
            dest[i] = float(val)

    return top


# ---------------------------------------------------------------------------
# Live calibration fetching via IQMClient (mirrors CalibrationDataManager)
# ---------------------------------------------------------------------------

def fetch_calibration_from_backend(backend) -> Dict[str, Dict[str, float]]:
    """Pull live calibration data from an IQM backend.

    Uses the same machinery as `iqm-qubit-selector`'s `CalibrationDataManager`
    when that library is installed; otherwise tries the IQMClient API
    directly. Returns the parsed-dict shape.
    """
    try:
        from iqm.qubit_selector.qubit_selector import CalibrationDataManager  # type: ignore
        return CalibrationDataManager().get_calibration_fidelities(backend)
    except Exception:
        pass
    try:
        from iqm.iqm_client import IQMClient  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not fetch live calibration: install iqm-qubit-selector or "
            "iqm-client[qiskit] (both ship a working IQMClient)."
        ) from e
    url = getattr(backend, "_client", None)
    if url is not None and hasattr(url, "_iqm_server_url"):
        url = url._iqm_server_url
    qc = getattr(backend, "name", None)
    if callable(qc):
        qc = qc()
    client = IQMClient(url, quantum_computer=qc)
    qms = client.get_quality_metric_set()
    raw = [{"dut_field": o.dut_field, "value": o.value} for o in qms.observations]
    return parse_calibration_json(raw)


# ---------------------------------------------------------------------------
# Chain (path) selection
# ---------------------------------------------------------------------------

@dataclass
class ChainSelection:
    chain: List[int]
    score: float
    reason: str

    def chain_names(self, top: "Topology") -> List[str]:
        return [top.name_of(q) for q in self.chain]


def find_best_chain_path(
    top: Topology,
    n: int,
    *,
    forbidden: Optional[Iterable[int]] = None,
    max_chains: int = 5000,
) -> ChainSelection:
    """DFS enumeration; pick the highest-fidelity-product path of length n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(top.nodes):
        raise ValueError(f"chain of length {n} > device size {len(top.nodes)}")
    forbidden_set = set(forbidden) if forbidden else set()
    nodes = [q for q in top.nodes if q not in forbidden_set]

    best: Optional[ChainSelection] = None
    explored = 0
    for start in nodes:
        stack = [(start, [start])]
        while stack:
            if explored >= max_chains:
                break
            cur, path = stack.pop()
            if len(path) == n:
                explored += 1
                s = top.chain_score(path)
                if best is None or s > best.score:
                    best = ChainSelection(
                        chain=list(path),
                        score=s,
                        reason="auto: highest-fidelity connected path",
                    )
                continue
            for nb in sorted(top.neighbours(cur)):
                if nb in path or nb in forbidden_set:
                    continue
                stack.append((nb, path + [nb]))
        if explored >= max_chains:
            break
    if best is None:
        raise ValueError(f"no connected path of length {n} in {top.name!r}")
    return best


def validate_user_chain(
    top: Topology,
    n: int,
    user_chain: Sequence,
) -> Optional[ChainSelection]:
    """Return a ChainSelection if `user_chain` is a valid path; else None.
    Accepts qubit names ('QB7') or 0/1-indexed integers.
    """
    coerced: List[int] = []
    for q in user_chain:
        if isinstance(q, str):
            qn = _parse_qubit_label(q)
            if qn is None or qn not in top.qubit_name_to_index:
                return None
            coerced.append(top.qubit_name_to_index[qn])
        elif isinstance(q, int):
            if q in top.adjacency:
                coerced.append(q)
            elif (q - 1) in top.adjacency:
                coerced.append(q - 1)
            else:
                return None
        else:
            return None
    if len(coerced) != n or not top.is_connected_chain(coerced):
        return None
    return ChainSelection(
        chain=coerced,
        score=top.chain_score(coerced),
        reason="user-supplied chain (validated)",
    )


def select_chain(
    top: Topology,
    n: int,
    user_chain: Optional[Sequence] = None,
    strategy: str = "auto",
) -> ChainSelection:
    """High-level entry point.

    strategy:
      * "user"         : require user_chain, error if invalid
      * "user_or_auto" : try user_chain, fall back to auto on failure
      * "auto"         : best calibration-weighted path via local DFS
      * "iqm_selector" : official IQM Qubit Selector — handled by the runner
    """
    if strategy == "user":
        if user_chain is None:
            raise ValueError("strategy='user' requires user_chain")
        sel = validate_user_chain(top, n, user_chain)
        if sel is None:
            raise ValueError(f"user chain {list(user_chain)} not valid in {top.name!r}")
        return sel
    if strategy == "user_or_auto":
        if user_chain is not None:
            sel = validate_user_chain(top, n, user_chain)
            if sel is not None:
                return sel
            print(
                f"  [chain] user-supplied chain {list(user_chain)} is not a "
                f"valid path of length {n} in {top.name!r}; falling back to auto"
            )
        return find_best_chain_path(top, n)
    if strategy == "auto":
        return find_best_chain_path(top, n)
    if strategy == "iqm_selector":
        raise ValueError(
            "strategy='iqm_selector' is dispatched by the runner with a real "
            "IQM backend; see qubit_selector_adapter.select_chain_iqm."
        )
    raise ValueError(f"unknown strategy {strategy!r}")
