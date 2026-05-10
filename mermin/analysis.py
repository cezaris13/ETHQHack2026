"""Aggregate experimental measurement counts into <M_n>.

Each circuit produced by `circuits.build_all_circuits` measures one
tensor-product observable. With our basis-rotation convention, the
+1 eigenvalue of A(alpha_i) corresponds to bit 0 and the -1 eigenvalue to
bit 1. Therefore the eigenvalue of the full product term on a single shot is

    parity = (-1) ** (number of 1s in the bitstring)

and <term> = mean of `parity` over shots. The Mermin expectation value is
then

    <M_n> = sum_pattern  coef(pattern) * <term(pattern)>.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math

from polynomial import Pattern, MerminProblem


def _parity_of(bitstring: str) -> int:
    """+1 if the bitstring has even Hamming weight, -1 if odd.

    Qiskit returns counts keys as little-endian strings (cbit 0 on the right);
    parity is order-independent so we don't need to flip them.
    """
    ones = bitstring.count("1")
    return 1 if (ones % 2 == 0) else -1


def expectation_from_counts(counts: Dict[str, int]) -> Tuple[float, float]:
    """Return (mean, standard error of the mean) of the Z..Z parity from counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0, math.inf
    mean = 0.0
    sq = 0.0
    for bs, c in counts.items():
        p = _parity_of(bs)
        mean += p * c
        sq += (p * p) * c  # p**2 == 1 always, but keep the form for clarity
    mean /= total
    sq /= total
    var = sq - mean * mean
    sem = math.sqrt(max(var, 0.0) / total)
    return mean, sem


@dataclass
class TermResult:
    pattern: Pattern
    coefficient: float
    expectation: float
    sem: float                       # statistical standard error per term
    counts: Dict[str, int]           # raw counts dict


@dataclass
class MerminResult:
    """Outcome of a full Mermin experiment."""
    problem: MerminProblem
    physical_qubits: List
    backend_name: str
    shots: int
    term_results: List[TermResult]
    chain_reason: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        return sum(t.coefficient * t.expectation for t in self.term_results)

    @property
    def value_sem(self) -> float:
        # propagate per-term SEMs as if independent (good enough for shots)
        return math.sqrt(
            sum((t.coefficient * t.sem) ** 2 for t in self.term_results)
        )

    @property
    def violates_classical(self) -> bool:
        return abs(self.value) > self.problem.classical_bound + self.value_sem

    @property
    def quantum_ratio(self) -> float:
        return abs(self.value) / self.problem.quantum_bound

    def summary(self) -> str:
        v = self.value
        sem = self.value_sem
        cb = self.problem.classical_bound
        qb = self.problem.quantum_bound
        sep = "-" * 64
        out = [
            sep,
            f"Mermin experiment  n={self.problem.n}",
            f"  backend                : {self.backend_name}",
            f"  qubit chain            : {self.physical_qubits}",
            f"  chain selection reason : {self.chain_reason}",
            f"  shots per term         : {self.shots}",
            f"  number of terms in M_n : {len(self.term_results)}",
            f"  classical bound        : {cb}",
            f"  quantum   bound        : {qb:.6f}",
            f"  ideal <M_n>_GHZ        : {self.problem.theoretical_value:+.6f}",
            f"  measured <M_n>         : {v:+.6f}  (+/- {sem:.4f})",
            f"  |M_n| / quantum bound  : {self.quantum_ratio:.4f}",
            f"  violates classical?    : "
            f"{'YES' if self.violates_classical else 'no '} "
            f"({abs(v):.4f} vs {cb})",
        ]
        for note in self.notes:
            out.append(f"  note                   : {note}")
        out.append(sep)
        return "\n".join(out)
