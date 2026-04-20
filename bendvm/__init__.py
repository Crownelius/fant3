"""
BendVM — a virtual machine that operates on programs while they are compressed.

Named for the Apollonian curvatures ("bends") that are its native state type.

Core insight (from Kocik, arxiv:2001.05866 and Graham-Lagarias-Mallows-Wilks-Yan
2003): the Apollonian group is a discrete subgroup of SO(3,1); its generators
are three 4×4 integer reflections. Tangency spinors reduce this to pairs of
SL(2, Z) matrices acting on 2D integer vectors.

BendVM is the VM where:
  - STATE      = a Pauli spinor s ∈ Z² (or, in quartet mode, 4 integer curvatures)
  - INSTRUCTION = an SL(2, Z) matrix (or, in quartet mode, an Apollonian 4×4)
  - PROGRAM     = a sequence of instructions

but crucially, the PROGRAM itself IS represented as a single matrix — the product
of its constituent instructions. Emitting an instruction = left-multiplying the
program matrix. Composing two programs = one matrix multiplication, independent
of their length.

Key guarantees:
  - Composition: O(1) regardless of program length  (one matrix multiply)
  - Execution:   O(1) after compilation               (one matrix-vector multiply)
  - Inversion:   O(1)                                 (matrix inverse, det = ±1)
  - Equivalence: O(1)                                 (matrix equality)
  - Determinism: exact integer arithmetic, no rounding

This is "operating while compressed" for the subclass of programs expressible
as words in SL(2, Z) (or the 4×4 Apollonian group) — a strict subset of Turing-
computable functions, but a surprisingly rich one (continued fractions, Fibonacci-
family linear recurrences, Apollonian packing traversal, modular arithmetic,
finite-automata transitions, any reversible integer linear map).

Name collision note: this BendVM is unrelated to github.com/ApolloVM/apollovm_dart
(a Dart/Java polyglot VM). That project chose the name for the Greek god;
this one takes it from Apollonius of Perga via Descartes' bends.
"""

from .core import BendVM, Matrix, Spinor, matmul, matvec, det
from .instructions import S, T, I, R, ST_generators, apollonian_reflections
from .programs import (
    fibonacci_program, euclid_gcd_program,
    power_program, continued_fraction_program,
)

__all__ = [
    # Core
    "BendVM", "Matrix", "Spinor", "matmul", "matvec", "det",
    # Standard generators
    "S", "T", "I", "R", "ST_generators", "apollonian_reflections",
    # Pre-built programs
    "fibonacci_program", "euclid_gcd_program",
    "power_program", "continued_fraction_program",
]
