"""
BendVM core — state, matrix, and the VM class.

Pure Python integer arithmetic (no numpy) so programs run at arbitrary precision.
Fibonacci step N produces a number with ~0.48·N digits; numpy int64 overflows
at N≈93, but Python ints never do. This VM is meant to push precision limits.
"""

from __future__ import annotations
from typing import List, Tuple, Union

# A 2×2 integer matrix is four ints in row-major order.
# A Spinor is a 2-int column vector.
# A 4×4 matrix is 16 ints in row-major order.

Matrix = Tuple[int, ...]   # length 4 (2x2) or 16 (4x4)
Spinor = Tuple[int, int]   # length 2


def _dim(M: Matrix) -> int:
    if len(M) == 4:
        return 2
    if len(M) == 16:
        return 4
    raise ValueError(f"Matrix length {len(M)} is neither 4 (2x2) nor 16 (4x4)")


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix product A @ B for 2x2 or 4x4 integer matrices."""
    n = _dim(A)
    if _dim(B) != n:
        raise ValueError("Matrix dim mismatch")
    C = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s
    return tuple(C)


def matvec(A: Matrix, v: Union[Spinor, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Matrix-vector product for 2x2 or 4x4 matrices."""
    n = _dim(A)
    if len(v) != n:
        raise ValueError(f"Vector dim {len(v)} != matrix dim {n}")
    out = []
    for i in range(n):
        s = 0
        for j in range(n):
            s += A[i * n + j] * v[j]
        out.append(s)
    return tuple(out)


def det(A: Matrix) -> int:
    """Exact integer determinant for 2x2 or 4x4."""
    n = _dim(A)
    if n == 2:
        return A[0] * A[3] - A[1] * A[2]
    # 4x4 via cofactor expansion along first row
    s = 0
    for j in range(4):
        # Minor with row 0 + col j removed
        minor = []
        for r in range(1, 4):
            for c in range(4):
                if c == j:
                    continue
                minor.append(A[r * 4 + c])
        minor = tuple(minor)
        # 3x3 determinant of minor
        a, b, c, d, e, f, g, h, i = minor
        det3 = (a * (e * i - f * h)
                - b * (d * i - f * g)
                + c * (d * h - e * g))
        sign = -1 if (j & 1) else 1
        s += sign * A[j] * det3
    return s


def inverse(A: Matrix) -> Matrix:
    """
    Integer inverse for 2x2 matrices with det = ±1 (i.e. elements of SL(2, Z)
    or GL(2, Z)). For a 2x2 matrix [[a, b], [c, d]] with determinant d_:
        A^-1 = (1/d_) * [[d, -b], [-c, a]]
    When d_ = ±1, this stays integer.
    """
    n = _dim(A)
    d_ = det(A)
    if n == 2:
        if abs(d_) != 1:
            raise ValueError(
                f"2x2 matrix has determinant {d_}; integer inverse requires det = ±1")
        a, b, c, dd = A
        if d_ == 1:
            return (dd, -b, -c, a)
        return (-dd, b, c, -a)
    # 4x4: use cofactor matrix divided by determinant; only works cleanly
    # when det = ±1 (which is true for elements of the Apollonian group).
    if abs(d_) != 1:
        raise ValueError(
            f"4x4 matrix has determinant {d_}; integer inverse requires det = ±1")
    # Compute cofactor matrix
    cof = [0] * 16
    for i in range(4):
        for j in range(4):
            minor = []
            for r in range(4):
                if r == i:
                    continue
                for c in range(4):
                    if c == j:
                        continue
                    minor.append(A[r * 4 + c])
            a, b, c, dd, e, f, g, h, ii = minor
            m_det = (a * (f * ii - g * h)
                     - b * (dd * ii - g * e)
                     + c * (dd * h - f * e))
            cof[j * 4 + i] = ((-1) ** (i + j)) * m_det  # note transpose
    # Adjugate / det
    result = tuple(x * d_ for x in cof)  # d_ = ±1, so * d_ = / d_
    return result


def identity(n: int) -> Matrix:
    """Identity matrix of dimension 2 or 4."""
    M = [0] * (n * n)
    for i in range(n):
        M[i * n + i] = 1
    return tuple(M)


# ─────────────────────────────────────────────────────────────────────────────
#  The VM itself
# ─────────────────────────────────────────────────────────────────────────────

class BendVM:
    """
    A virtual machine that represents a program as a single matrix (the product
    of its instruction matrices) and a state as a vector.

    Typical usage:

        vm = BendVM(initial_state=(1, 0))
        for step in program:
            vm.emit(step)             # accumulates into vm.program matrix
        final_state = vm.run()        # applies program matrix to initial state

    But the key move is that `emit` accumulates into a single matrix, so the
    entire million-step program is stored in 4 integers (2x2) or 16 integers
    (4x4), regardless of program length.
    """

    def __init__(self, initial_state=(1, 0), dim: int = 2):
        if dim not in (2, 4):
            raise ValueError("BendVM supports dim=2 (SL(2,Z)) or dim=4 (Apollonian O(3,1))")
        if len(initial_state) != dim:
            raise ValueError(f"initial_state length {len(initial_state)} != dim {dim}")
        self.dim = dim
        self.initial_state = tuple(int(x) for x in initial_state)
        self.program: Matrix = identity(dim)
        self.n_instructions = 0   # counts for reporting only; doesn't affect storage

    def emit(self, matrix: Matrix) -> "BendVM":
        """Append an instruction. The program matrix accumulates in place.
        No intermediate state is materialized."""
        if _dim(matrix) != self.dim:
            raise ValueError(f"Instruction dim {_dim(matrix)} != VM dim {self.dim}")
        self.program = matmul(matrix, self.program)
        self.n_instructions += 1
        return self

    def emit_many(self, matrices) -> "BendVM":
        """Emit a sequence of instructions."""
        for m in matrices:
            self.emit(m)
        return self

    def run(self) -> Tuple[int, ...]:
        """Apply the composed program matrix to the initial state. O(1)
        regardless of how many instructions were emitted."""
        return matvec(self.program, self.initial_state)

    def compose(self, other: "BendVM") -> "BendVM":
        """Return a new VM whose program is `other` applied AFTER `self`.
        One matrix multiplication. Instruction counts add for reporting."""
        if other.dim != self.dim:
            raise ValueError("Cannot compose VMs of different dimension")
        out = BendVM(self.initial_state, dim=self.dim)
        out.program = matmul(other.program, self.program)
        out.n_instructions = self.n_instructions + other.n_instructions
        return out

    def invert(self) -> "BendVM":
        """Return a new VM whose program is the inverse of self's program.
        Exact — no precision loss — because det = ±1 on the Apollonian group."""
        out = BendVM(self.initial_state, dim=self.dim)
        out.program = inverse(self.program)
        out.n_instructions = self.n_instructions  # inverse has same "size"
        return out

    def equals(self, other: "BendVM") -> bool:
        """Two programs are equivalent iff their matrices are equal. O(1)."""
        return self.dim == other.dim and self.program == other.program

    def power(self, n: int) -> "BendVM":
        """Run the program n times via fast matrix exponentiation. O(log n)
        matrix multiplies, so a million repetitions take ~20 multiplies."""
        if n < 0:
            return self.invert().power(-n)
        out = BendVM(self.initial_state, dim=self.dim)
        base = self.program
        result = identity(self.dim)
        while n > 0:
            if n & 1:
                result = matmul(base, result)
            base = matmul(base, base)
            n >>= 1
        out.program = result
        return out

    # Introspection helpers ---------------------------------------------------

    def state(self) -> Tuple[int, ...]:
        """Alias for run()."""
        return self.run()

    def bits(self) -> int:
        """How many bits does the compressed program matrix take?
        Sum of bit-lengths of the integer entries."""
        return sum(x.bit_length() for x in self.program) + self.dim * self.dim

    def __repr__(self):
        n = self.dim
        rows = []
        for i in range(n):
            rows.append("[" + " ".join(str(self.program[i * n + j]) for j in range(n)) + "]")
        return (f"BendVM(dim={n}, n_instructions={self.n_instructions}, "
                f"state={self.initial_state}, program=[{' '.join(rows)}])")
