# `mermin_iqm` — Generalized Mermin inequality test for IQM hardware

Implementation of the protocol from

> P. De Fabritiis, I. Roditi, S.P. Sorella,
> *"Mermin's inequalities in Quantum Field Theory"*,
> Phys. Lett. B **846** (2023) 138198 — [arXiv:2305.04546](https://arxiv.org/abs/2305.04546).

for **arbitrary n** on real **IQM Crystal-class** quantum processors
(Garnet, Emerald, ...) accessed through IQM Resonance.

The runner falls back gracefully when the requested setup is not realisable
on the device:
*user-supplied chain → hand-crafted chain → auto-selected chain → AerSimulator.*

---

## What's in the box

```
mermin_iqm/
├── __init__.py        public API
├── polynomial.py      M_n recursion (paper Eq. 1) + analytic GHZ value + angle optimization
├── topology.py        Crystal-20 / Crystal-54 graphs, calibration loader, chain selector
├── circuits.py        GHZ prep + measurement-basis rotation circuits (qiskit)
├── analysis.py        counts → ⟨M_n⟩ aggregation
└── runner.py          end-to-end orchestration with fallback handling

mermin_iqm_demo.ipynb  walkthrough notebook (executable end-to-end on a simulator)
```

## Quick start

```python
from mermin_iqm import run_mermin_test

# Simulator fallback — no IQM credentials needed
result = run_mermin_test(n=3, shots=8192)
print(result.summary())
```

```text
Mermin experiment  n=3
  backend                : AerSimulator
  qubit chain            : [0, 1, 2]
  shots per term         : 8192
  number of terms in M_n : 4
  classical bound        : 2.0
  quantum   bound        : 4.000000
  ideal <M_n>_GHZ        : +4.000000
  measured <M_n>         : +4.000000  (+/- 0.0000)
  |M_n| / quantum bound  : 1.0000
  violates classical?    : YES (4.0000 vs 2.0)
```

## Running on real IQM hardware

```python
result = run_mermin_test(
    n=4,
    iqm_server_url="https://cocos.resonance.meetiqm.com/garnet",
    iqm_token="...",
    iqm_device="garnet",                 # or "emerald"
    chain_strategy="hand_or_auto",
    calibration_json="callibration.json",  # downloaded from Resonance
    shots=8192,
)
print(result.summary())
```

The runner picks up the live coupling graph from `backend.coupling_map`,
weighs candidate chains by 1q + 2q gate fidelities from the calibration
JSON, and validates connectivity before submission.

---

## Math: where the implementation comes from

### 1. The Mermin polynomial (paper Eq. 1)

Recursion:

> M_n = ½ M_{n-1} (A_n + A'_n) + ½ M'_{n-1} (A_n − A'_n)
> M_1 = 2 A_1

`mermin_polynomial(n)` returns `{primed_pattern: coefficient}` for the
2^n possible patterns; about half cancel for odd n (paper Eq. 3 for n=3).

### 2. Optimal angles (paper Sec. 2)

Each operator is parameterized as

> A(α) = cos(α) X + sin(α) Y

The analytic GHZ formula is

> ⟨A(γ_1) ⊗ ... ⊗ A(γ_n)⟩_GHZ = cos(γ_1 + ... + γ_n)

The paper gives optimal angles for n=3, 4. The package uses the
generalized scheme

* α_1 = 0, β_1 = π/2
* α_i = −π/4, β_i = +π/4 for i ≥ 2

which `polynomial.standard_optimal_angles(n)` exposes. We verify
numerically that this scheme **saturates the QM bound 2^((n+1)/2)** for
every n ≤ 10. A scipy-based numerical optimizer is also available
(`optimize_angles(n)`) for sanity checking or for non-standard variants.

### 3. Measurement protocol (paper Sec. 5)

To measure A(α) we use the identity

> A(α) = R_z(−α) X R_z(α) = R_z(−α) H Z H R_z(α)

so applying R_z(α) followed by H to the qubit before computational-basis
measurement gives a 0/1 outcome encoding the +1/−1 eigenvalue of A(α).
The eigenvalue of the n-fold product is the **parity** (−1)^(# of 1s) of
the bitstring.

### 4. GHZ preparation

Standard linear construction: H on chain[0], then CX(chain[i],
chain[i+1]) along the n-qubit chain. This needs a **connected path** of
length n in the coupling graph, which is what the chain selector
guarantees.

---

## Hardware-failure handling

| trigger                                        | response                           |
|------------------------------------------------|-------------------------------------|
| `n` larger than device                         | hard `ValueError` (early)           |
| `n > 20`                                       | hard `ValueError` (2^n circuits)    |
| user-supplied chain not connected (`user_or_auto`) | warning + auto-select        |
| user-supplied chain not connected (`user`)     | hard `ValueError`                   |
| no hand-crafted chain for this length (`hand_or_auto`) | auto-select                |
| no hand-crafted chain (`hand`)                 | hard `ValueError`                   |
| backend `coupling_map` unreadable              | warning + Crystal-N default graph  |
| transpile with `initial_layout` fails          | retry without explicit layout       |
| no IQM credentials                             | `AerSimulator` fallback             |

The auto-selector scores a candidate path by

> score = Π_i F_1q(qubit_i) · Π_(a,b) F_2q(edge a-b)

with missing fidelities defaulting to 0.99 so that calibration coverage
gaps don't bias the search. Hand-crafted chains for Crystal-20 and
Crystal-54 are stored in `topology.HAND_CRAFTED_CHAINS` and are easy to
override per experiment.

---

## Verified end-to-end on the simulator

Sweep n = 2 ... 8 (4 096 shots / term):

| n | measured \|⟨M_n⟩\|     | classical bound | quantum bound 2^((n+1)/2) |
|---|--------------------|-----------------|--------------------------|
| 2 | 2.83 ± 0.02       | 2.0             | 2.83                    |
| 3 | 4.00 ± 0.00       | 2.0             | 4.00                    |
| 4 | 5.66 ± 0.02       | 2.0             | 5.66                    |
| 5 | 8.00 ± 0.00       | 2.0             | 8.00                    |
| 6 | 11.31 ± 0.02      | 2.0             | 11.31                   |
| 7 | 16.00 ± 0.00      | 2.0             | 16.00                   |
| 8 | 22.62 ± 0.03      | 2.0             | 22.63                   |

Odd-n rows show *zero* statistical uncertainty because every nonzero
term has an exactly ±1 expectation on GHZ with the chosen angles, so all
shots produce the same parity. Even-n rows have small statistical
fluctuation around the analytic value.

---

## Limits

* Each Mermin term costs one circuit, so **n=20** is the practical
  ceiling: 2^20 ≈ 10^6 circuits is already too many for a Resonance run.
  For comparing across n, n ≤ 10 is comfortable.
* The default Crystal-20 / Crystal-54 graphs are a reasonable
  approximation but **not authoritative**; on a real run the graph is
  pulled from `backend.coupling_map`.
* Readout error mitigation, twirling, dynamical decoupling, ZNE, ...
  are *not* applied; this implementation reproduces the raw protocol.
  The `term_results` field of `MerminResult` exposes per-term counts
  so users can plug in their preferred mitigation pipeline.

---

## Dependencies

* qiskit ≥ 1.0
* qiskit-aer (for the simulator fallback)
* qiskit-iqm — only needed for real-hardware runs
* numpy, scipy (scipy is optional; falls back to coordinate descent)
* matplotlib — only for the demo notebook
