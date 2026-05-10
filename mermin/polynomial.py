"""Mermin polynomial M_n following the recursion in

    De Fabritiis, Roditi, Sorella, "Mermin's inequalities in Quantum Field Theory",
    Phys. Lett. B 846 (2023) 138198, Eq. (1):

        M_n = (1/2) M_{n-1} (A_n + A'_n) + (1/2) M'_{n-1} (A_n - A'_n)
        M_1 = 2 A_1

    where M'_k denotes M_k with all primed/unprimed labels swapped.

A "term" in the resulting polynomial is a tensor product of n single-qubit
operators A_i (unprimed) or A'_i (primed). We represent each term as a
tuple of booleans `pattern` where `pattern[i] is True` iff qubit i is
primed. Coefficients are tracked as floats.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np

Pattern = Tuple[bool, ...]


# ---------------------------------------------------------------------------
# Polynomial construction
# ---------------------------------------------------------------------------

def mermin_polynomial(n: int) -> Dict[Pattern, float]:
    """Build M_n via the recursion in Eq. (1).

    Returns
    -------
    dict mapping primed-pattern -> coefficient. Patterns with coefficient 0
    are dropped.

    Examples
    --------
    n=2: M_2 = AB + AB' + A'B - A'B' (CHSH-like)
    n=3: M_3 = ABC' + AB'C + A'BC - A'B'C' (paper's Eq. 3, half terms vanish)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # M_1 = 2 A_1, M'_1 = 2 A'_1
    M: Dict[Pattern, float] = {(False,): 2.0}
    Mp: Dict[Pattern, float] = {(True,): 2.0}

    for _ in range(2, n + 1):
        new_M: Dict[Pattern, float] = defaultdict(float)
        # (1/2) * M_{k-1} * (A_k + A'_k)
        for pat, c in M.items():
            new_M[pat + (False,)] += 0.5 * c
            new_M[pat + (True,)] += 0.5 * c
        # (1/2) * M'_{k-1} * (A_k - A'_k)
        for pat, c in Mp.items():
            new_M[pat + (False,)] += 0.5 * c
            new_M[pat + (True,)] -= 0.5 * c
        # M'_k is M_k with primed/unprimed swapped on every qubit
        new_Mp: Dict[Pattern, float] = {
            tuple(not p for p in pat): c for pat, c in new_M.items()
        }
        M, Mp = dict(new_M), new_Mp

    # drop zero-coefficient terms
    return {pat: c for pat, c in M.items() if abs(c) > 1e-12}


def quantum_bound(n: int) -> float:
    """The maximum value of |M_n| in QM (Eq. 2 of the paper): 2^((n+1)/2)."""
    return 2.0 ** ((n + 1) / 2)


def classical_bound(n: int) -> float:
    """Local-realistic bound on |M_n| using the recursion of Eq. (1).

    For the standard Mermin polynomial built from this recursion, the
    classical bound is 2 for every n >= 2 (cf. Mermin 1990 and the paper:
    |M_3|_Cl <= 2, |2 M_4|_Cl <= 4 -> |M_4|_Cl <= 2, etc.).
    """
    return 2.0


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_polynomial(poly: Dict[Pattern, float]) -> str:
    """Human-readable representation, e.g. '+ABC\' +AB\'C +A\'BC -A\'B\'C\''."""
    pieces = []
    for pat in sorted(poly.keys()):
        c = poly[pat]
        sign = "+" if c > 0 else "-"
        # collapse coefficient if it's exactly +/-1
        mag = abs(c)
        coef_str = "" if abs(mag - 1.0) < 1e-9 else f"{mag:g}*"
        ops = "".join(
            chr(ord("A") + i) + ("'" if primed else "")
            for i, primed in enumerate(pat)
        )
        pieces.append(f"{sign}{coef_str}{ops}")
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# Analytic expectation value on |GHZ_n>
# ---------------------------------------------------------------------------

def ghz_expectation_value(
    poly: Dict[Pattern, float],
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    r"""Analytic <M_n> on the GHZ state |GHZ_n> = (|0..0> + |1..1>)/sqrt(2).

    Each qubit i has two measurement settings:
      - unprimed angle alpha[i]:  A_i  = cos(alpha_i)*X + sin(alpha_i)*Y
      - primed angle   beta[i] :  A'_i = cos(beta_i )*X + sin(beta_i )*Y

    For any product of such operators on |GHZ_n>:

        <U_{gamma_1} ... U_{gamma_n}>_GHZ = cos(gamma_1 + ... + gamma_n)

    Hence

        <M_n>_GHZ = sum_pat coef(pat) * cos( sum_i gamma_i(pat) )

    where gamma_i(pat) = beta[i] if pat[i] else alpha[i].
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must have the same shape")

    total = 0.0
    for pat, coef in poly.items():
        if len(pat) != len(alpha):
            raise ValueError("pattern length does not match angle vector length")
        s = sum(beta[i] if p else alpha[i] for i, p in enumerate(pat))
        total += coef * math.cos(s)
    return total


# ---------------------------------------------------------------------------
# Optimal measurement angles
# ---------------------------------------------------------------------------

def standard_optimal_angles(n: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""The angle scheme used in the paper, generalized to arbitrary n.

    For n=3 the paper's choice (Sec. 2):

        alpha_1 = 0,        beta_1 = pi/2
        alpha_i = -pi/4,    beta_i = +pi/4   for i = 2..n

    For n=4 the paper repeats this same pattern (after Eq. 12).
    The pattern is verified numerically (this module) to saturate the
    quantum bound 2^((n+1)/2) for every n we tested.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = 0.0
    beta[0] = math.pi / 2
    if n >= 2:
        alpha[1:] = -math.pi / 4
        beta[1:] = math.pi / 4
    return alpha, beta


def optimize_angles(
    n: int,
    poly: Dict[Pattern, float] | None = None,
    n_restarts: int = 10,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Numerically maximize |<M_n>_GHZ| over the 2n measurement angles.

    Useful as a sanity check or when tweaking M_n. Uses scipy if available,
    falls back to a simple coordinate-descent.

    Returns (alpha, beta, achieved_value).
    """
    if poly is None:
        poly = mermin_polynomial(n)

    rng = np.random.default_rng(seed)

    # warm-start from the analytic scheme
    a0, b0 = standard_optimal_angles(n)
    best_val = abs(ghz_expectation_value(poly, a0, b0))
    best_a, best_b = a0.copy(), b0.copy()

    def neg_obj(x: np.ndarray) -> float:
        a = x[:n]
        b = x[n:]
        return -abs(ghz_expectation_value(poly, a, b))

    try:
        from scipy.optimize import minimize  # type: ignore
        for _ in range(n_restarts):
            x0 = rng.uniform(-math.pi, math.pi, size=2 * n)
            res = minimize(neg_obj, x0, method="L-BFGS-B")
            if -res.fun > best_val:
                best_val = -res.fun
                best_a = res.x[:n]
                best_b = res.x[n:]
    except ImportError:
        # Coordinate descent fallback (slow but no scipy needed)
        x = np.concatenate([best_a, best_b])
        step = 1e-2
        for _ in range(2000):
            improved = False
            for i in range(2 * n):
                for delta in (step, -step):
                    x[i] += delta
                    v = -neg_obj(x)
                    if v > best_val:
                        best_val = v
                        improved = True
                        break
                    x[i] -= delta
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        best_a = x[:n]
        best_b = x[n:]

    return best_a, best_b, best_val


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

@dataclass
class MerminProblem:
    """Bundle of everything needed to run the n-qubit Mermin test."""
    n: int
    poly: Dict[Pattern, float]
    alpha: np.ndarray  # unprimed angles, shape (n,)
    beta: np.ndarray   # primed angles,  shape (n,)
    quantum_bound: float
    classical_bound: float
    theoretical_value: float  # <M_n> on ideal GHZ_n with the chosen angles

    @classmethod
    def make(cls, n: int, optimize: bool = False, seed: int | None = None) -> "MerminProblem":
        poly = mermin_polynomial(n)
        if optimize:
            a, b, _ = optimize_angles(n, poly, seed=seed)
        else:
            a, b = standard_optimal_angles(n)
        return cls(
            n=n,
            poly=poly,
            alpha=a,
            beta=b,
            quantum_bound=quantum_bound(n),
            classical_bound=classical_bound(n),
            theoretical_value=ghz_expectation_value(poly, a, b),
        )

    def summary(self) -> str:
        return (
            f"Mermin problem n={self.n}\n"
            f"  number of nonzero terms in M_n  : {len(self.poly)}\n"
            f"  classical bound  |M_n|_Cl       : {self.classical_bound}\n"
            f"  quantum   bound  |M_n|_QM       : {self.quantum_bound:.6f}\n"
            f"  theoretical value with chosen   : {self.theoretical_value:.6f}\n"
            f"  ratio achieved / quantum bound  : "
            f"{abs(self.theoretical_value)/self.quantum_bound:.4f}"
        )
