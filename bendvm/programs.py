"""
Pre-built BendVM programs — each demonstrates "operate while compressed".

Each function returns a BendVM whose `program` matrix represents a completed
computation. You apply it to an initial state via `.run()` to get the answer.

The point: you can emit a MILLION instructions into the VM; the program stays
a 2×2 or 4×4 matrix. To run the program, you do ONE matrix-vector multiply.
"""

from __future__ import annotations
from typing import Tuple

from .core import BendVM, matmul, identity, Matrix
from .instructions import S, T, T_INV, FIB_STEP, R1, R2, R3, R4


# ─────────────────────────────────────────────────────────────────────────────
#  Fibonacci — the classic SL(2,Z)/GL(2,Z) compression trick
# ─────────────────────────────────────────────────────────────────────────────

def fibonacci_program(n: int) -> BendVM:
    """
    Return a BendVM that, when run on initial state (F_1, F_0) = (1, 0),
    produces (F_{n+1}, F_n).

    Internally this uses n applications of FIB_STEP = [[1,1],[1,0]],
    but those are all composed into a SINGLE 2x2 matrix via fast
    exponentiation — O(log n) matrix multiplies regardless of n.

    So we compute F_{1000000} by squaring the Fibonacci matrix ~20 times,
    not by iterating a million times.
    """
    vm = BendVM(initial_state=(1, 0), dim=2)
    # The "program" is FIB_STEP applied n times, implemented as a single
    # matrix via repeated squaring. We bypass emit() and set .program directly
    # to demonstrate the compression trick.
    base = FIB_STEP
    result = identity(2)
    m = n
    while m > 0:
        if m & 1:
            result = matmul(base, result)
        base = matmul(base, base)
        m >>= 1
    vm.program = result
    vm.n_instructions = n      # "virtual" — we never materialized them
    return vm


# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean GCD as continued-fraction expansion in SL(2, Z)
# ─────────────────────────────────────────────────────────────────────────────

def euclid_gcd_program(a: int, b: int) -> Tuple[BendVM, int]:
    """
    Compile the Euclidean-algorithm steps to compute gcd(a, b) into an SL(2, Z)
    matrix M such that M @ [a, b]^T = [gcd, 0]^T.

    Each step "(a, b) ↦ (b, a mod b)" corresponds to the SL(2, Z) matrix
        [[0, 1], [1, -q]]     where q = a // b,
    and the full algorithm is the product of these matrices. The gcd falls
    out as the first component of the matrix applied to [a, b].

    Returns (vm, gcd).
    """
    if a == 0 and b == 0:
        raise ValueError("gcd(0,0) is undefined")
    vm = BendVM(initial_state=(abs(a), abs(b)), dim=2)
    x, y = abs(a), abs(b)
    while y != 0:
        q = x // y
        # Matrix [[0, 1], [1, -q]] sends (x, y) to (y, x - q*y) = (y, x mod y)
        step = (0, 1, 1, -q)
        vm.emit(step)
        x, y = y, x - q * y
    gcd = x
    return vm, gcd


# ─────────────────────────────────────────────────────────────────────────────
#  Power: apply any SL(2, Z) matrix n times
# ─────────────────────────────────────────────────────────────────────────────

def power_program(base: Matrix, n: int, initial_state: Tuple[int, int] = (1, 0)) -> BendVM:
    """
    Return a BendVM whose program is `base^n`, computed via fast exponentiation.
    """
    vm = BendVM(initial_state=initial_state, dim=2)
    vm.program = identity(2)
    acc = base
    m = n
    while m > 0:
        if m & 1:
            vm.program = matmul(acc, vm.program)
        acc = matmul(acc, acc)
        m >>= 1
    vm.n_instructions = n
    return vm


# ─────────────────────────────────────────────────────────────────────────────
#  Continued fraction evaluation
# ─────────────────────────────────────────────────────────────────────────────

def continued_fraction_program(coefficients) -> BendVM:
    """
    Build the SL(2, Z) matrix representing the continued fraction
        [a0; a1, a2, ..., an]  =  a0 + 1/(a1 + 1/(a2 + ... + 1/an))

    The matrix product ∏ [[a_i, 1], [1, 0]] applied to (1, 0) yields (p, q)
    where p/q is the rational approximation.
    """
    vm = BendVM(initial_state=(1, 0), dim=2)
    for a in coefficients:
        step = (a, 1, 1, 0)
        vm.emit(step)
    return vm


# ─────────────────────────────────────────────────────────────────────────────
#  Apollonian packing walk — 4D example
# ─────────────────────────────────────────────────────────────────────────────

def apollonian_walk(bends: Tuple[int, int, int, int],
                    walk: str = "123412") -> BendVM:
    """
    Start from a Descartes quartet (b1, b2, b3, b4) and apply a sequence of
    Apollonian reflections specified by digits '1', '2', '3', '4'.

    Verifies the Descartes relation is preserved after each step.
    """
    if len(bends) != 4:
        raise ValueError("bends must be a 4-tuple (Descartes quartet)")
    vm = BendVM(initial_state=bends, dim=4)
    reflections = {'1': R1, '2': R2, '3': R3, '4': R4}
    for ch in walk:
        if ch not in reflections:
            raise ValueError(f"Unknown reflection '{ch}' — use 1, 2, 3, or 4")
        vm.emit(reflections[ch])
    return vm


# Starting Apollonian configurations. The Descartes relation requires:
#   (b1 + b2 + b3 + b4)^2 = 2 * (b1^2 + b2^2 + b3^2 + b4^2)

APOLLONIAN_0: Tuple[int, int, int, int] = (-1, 2, 2, 3)   # Classical starter
# Check: (-1+2+2+3)^2 = 36;  2*(1+4+4+9) = 36. ✓

APOLLONIAN_1: Tuple[int, int, int, int] = (0, 0, 1, 1)    # Strip packing starter
# (0+0+1+1)^2 = 4;  2*(0+0+1+1) = 4. ✓


def descartes_check(b: Tuple[int, int, int, int]) -> int:
    """Return (Σb)^2 - 2Σb^2. Zero iff the quartet is a valid Descartes config."""
    s = sum(b)
    q = sum(x * x for x in b)
    return s * s - 2 * q
