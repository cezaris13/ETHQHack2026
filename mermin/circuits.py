"""Qiskit circuits for Mermin's inequality experiment.

For each term in the Mermin polynomial M_n we build one circuit:

  1. Prepare |GHZ_n>  on the chosen physical qubits via H + CX-chain.
  2. Rotate qubit i into the eigenbasis of the operator we want to measure
     (A_i if pattern[i] is False, A'_i if True). The rotation is

         A_alpha = cos(alpha) X + sin(alpha) Y
                 = R_z(-alpha) X R_z(+alpha)
                 = R_z(-alpha) H Z H R_z(+alpha)

     so measuring A_alpha is equivalent to applying R_z(alpha) followed by
     H and measuring Z.
  3. Measure all n qubits in the computational basis.

We map *abstract* qubit indices 0..n-1 (chain position) onto the *physical*
hardware qubits in the chosen chain via Qiskit's `initial_layout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Dict

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from polynomial import Pattern


def ghz_state_circuit(n: int) -> QuantumCircuit:
    if n < 1:
        raise ValueError("n must be >= 1")
    qc = QuantumCircuit(n, name=f"GHZ_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def measurement_basis_change(qc: QuantumCircuit, qubit: int, angle: float) -> None:
    """Rotate `qubit` so that A(angle) = cos(a)X+sin(a)Y is mapped onto Z.

    Concretely: apply R_z(angle) then H. After this, a 0/1 outcome in the
    computational basis encodes the +1/-1 eigenvalue of A(angle).
    """
    if abs(angle) > 1e-12:
        qc.rz(angle, qubit)
    qc.h(qubit)


def mermin_term_circuit(
    n: int,
    pattern: Pattern,
    alpha: np.ndarray,
    beta: np.ndarray,
    *,
    name: str | None = None,
) -> QuantumCircuit:
    """Build one circuit corresponding to a single term in M_n.

    `pattern[i]` selects beta[i] (primed) when True, alpha[i] otherwise.
    """
    if len(pattern) != n:
        raise ValueError("pattern length mismatch")
    if alpha.shape != (n,) or beta.shape != (n,):
        raise ValueError("alpha/beta must have shape (n,)")

    qc = ghz_state_circuit(n)
    qc.barrier()
    for i, primed in enumerate(pattern):
        angle = beta[i] if primed else alpha[i]
        measurement_basis_change(qc, i, angle)
    qc.barrier()
    creg = ClassicalRegister(n, name="c")
    qc.add_register(creg)
    qc.measure(range(n), range(n))
    qc.name = name or _term_name(pattern)
    return qc


def _term_name(pattern: Pattern) -> str:
    return "term_" + "".join("Y" if p else "X" for p in pattern)


@dataclass
class CompiledMerminCircuits:
    """Bundle of all circuits + bookkeeping for a Mermin run."""
    n: int
    patterns: List[Pattern]
    coefficients: List[float]
    circuits: List[QuantumCircuit]
    physical_qubits: List  # physical qubit chain (length n), in order
    initial_layout: Dict[int, int]  # virtual -> physical (qiskit transpile arg)


def build_all_circuits(
    n: int,
    poly: Dict[Pattern, float],
    alpha: np.ndarray,
    beta: np.ndarray,
    physical_qubits: Sequence,
) -> CompiledMerminCircuits:
    """Make one circuit per nonzero term in M_n.

    `physical_qubits` is the chain produced by the qubit selector. Its
    length must equal n. The initial-layout dict maps virtual qubit i
    (chain position i) to the physical qubit at that position.
    """
    if len(physical_qubits) != n:
        raise ValueError(
            f"physical_qubits has length {len(physical_qubits)} but n={n}"
        )

    patterns: List[Pattern] = []
    coefs: List[float] = []
    circuits: List[QuantumCircuit] = []

    for pat, coef in sorted(poly.items()):
        patterns.append(pat)
        coefs.append(coef)
        circuits.append(mermin_term_circuit(n, pat, alpha, beta))

    # Convert physical qubit labels to integer indices for Qiskit's
    # initial_layout. Topology is now always 0-indexed (matches qiskit),
    # so the conversion is just a passthrough.
    layout: Dict[int, int] = {}
    for virtual, phys in enumerate(physical_qubits):
        layout[virtual] = int(phys)

    return CompiledMerminCircuits(
        n=n,
        patterns=patterns,
        coefficients=coefs,
        circuits=circuits,
        physical_qubits=list(physical_qubits),
        initial_layout=layout,
    )
