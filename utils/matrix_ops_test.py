"""
Tests for utils/matrix_ops.py

Run from the project root with:
    python -m utils.matrix_ops_test
"""

import sys
import os

# ── make sure the project root is on sys.path ──────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.matrix_ops import (
    vector_add,
    vector_subtract,
    scalar_vector_multiply,
    dot_product,
    hadamard_product,
    vector_sum,
    vector_mean,
    argmax,
    matrix_vector_multiply,
    outer_product,
    matrix_transpose,
    matrix_add,
    scalar_matrix_multiply,
)

import numpy as np  # used only to verify expected values

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

PASS = "  PASS"
FAIL = "  FAIL"
_failures = []


def _almost_equal(a, b, tol=1e-9):
    """Return True if two scalars are within tol of each other."""
    return abs(a - b) < tol


def _list_almost_equal(l1, l2, tol=1e-9):
    """Return True if every corresponding pair of elements is within tol."""
    if len(l1) != len(l2):
        return False
    return all(_almost_equal(x, y, tol) for x, y in zip(l1, l2))


def _matrix_almost_equal(m1, m2, tol=1e-9):
    """Return True if every row of two matrices is element-wise within tol."""
    if len(m1) != len(m2):
        return False
    return all(_list_almost_equal(r1, r2, tol) for r1, r2 in zip(m1, m2))


def _run(test_name: str, passed: bool):
    """Print result and record any failures."""
    status = PASS if passed else FAIL
    print(f"{status}  {test_name}")
    if not passed:
        _failures.append(test_name)


def _expect_error(func, args, exc_type, test_name: str):
    """
    Assert that func(*args) raises exc_type.
    Records PASS/FAIL via _run.
    """
    try:
        func(*args)
        _run(test_name, False)          # no exception → failure
    except exc_type:
        _run(test_name, True)
    except Exception as e:
        print(f"  FAIL  {test_name}  (unexpected exception: {e})")
        _failures.append(test_name)


# ─────────────────────────────────────────────
# vector_add
# ─────────────────────────────────────────────

def test_vector_add():
    print("\n--- vector_add ---")

    # basic integer vectors
    result = vector_add([1, 2, 3], [4, 5, 6])
    _run("vector_add basic integers", result == [5, 7, 9])

    # float vectors
    result = vector_add([0.1, 0.2], [0.3, 0.4])
    _run("vector_add floats", _list_almost_equal(result, [0.4, 0.6]))

    # single-element vectors
    _run("vector_add single element", vector_add([7], [-3]) == [4])

    # zeros
    _run("vector_add with zeros", vector_add([0, 0, 0], [1, 2, 3]) == [1, 2, 3])

    # negative values
    _run("vector_add negatives", vector_add([-1, -2], [-3, -4]) == [-4, -6])

    # shape mismatch
    _expect_error(vector_add, [[1, 2], [3]], ValueError, "vector_add length mismatch")

    # wrong type for v1
    _expect_error(vector_add, ["not_a_list", [1, 2]], TypeError, "vector_add wrong type v1")

    # wrong type for v2
    _expect_error(vector_add, [[1, 2], "not_a_list"], TypeError, "vector_add wrong type v2")


# ─────────────────────────────────────────────
# vector_subtract
# ─────────────────────────────────────────────

def test_vector_subtract():
    print("\n--- vector_subtract ---")

    _run("vector_subtract basic", vector_subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3])
    _run("vector_subtract floats",
         _list_almost_equal(vector_subtract([1.5, 2.5], [0.5, 1.5]), [1.0, 1.0]))
    _run("vector_subtract negative result",
         vector_subtract([1, 2], [3, 4]) == [-2, -2])
    _run("vector_subtract self gives zeros",
         vector_subtract([3, 3, 3], [3, 3, 3]) == [0, 0, 0])

    _expect_error(vector_subtract, [[1, 2], [1]], ValueError, "vector_subtract length mismatch")
    _expect_error(vector_subtract, [42, [1, 2]], TypeError, "vector_subtract wrong type v1")
    _expect_error(vector_subtract, [[1, 2], 42], TypeError, "vector_subtract wrong type v2")


# ─────────────────────────────────────────────
# scalar_vector_multiply
# ─────────────────────────────────────────────

def test_scalar_vector_multiply():
    print("\n--- scalar_vector_multiply ---")

    _run("scalar_vector_multiply by 2",
         scalar_vector_multiply(2, [1, 2, 3]) == [2, 4, 6])
    _run("scalar_vector_multiply by 0",
         scalar_vector_multiply(0, [1, 2, 3]) == [0, 0, 0])
    _run("scalar_vector_multiply by -1",
         scalar_vector_multiply(-1, [1, -2, 3]) == [-1, 2, -3])
    _run("scalar_vector_multiply float scalar",
         _list_almost_equal(scalar_vector_multiply(0.5, [2.0, 4.0]), [1.0, 2.0]))
    _run("scalar_vector_multiply single element",
         scalar_vector_multiply(3, [7]) == [21])

    _expect_error(scalar_vector_multiply, ["a", [1, 2]], TypeError,
                  "scalar_vector_multiply non-numeric scalar")
    _expect_error(scalar_vector_multiply, [2, "not_a_list"], TypeError,
                  "scalar_vector_multiply non-list vector")


# ─────────────────────────────────────────────
# dot_product
# ─────────────────────────────────────────────

def test_dot_product():
    print("\n--- dot_product ---")

    _run("dot_product basic", _almost_equal(dot_product([1, 2, 3], [4, 5, 6]), 32))
    _run("dot_product with zero vector",
         _almost_equal(dot_product([1, 2, 3], [0, 0, 0]), 0))
    _run("dot_product unit vectors",
         _almost_equal(dot_product([1, 0, 0], [1, 0, 0]), 1))
    _run("dot_product negative values",
         _almost_equal(dot_product([-1, -2], [3, 4]), -11))
    _run("dot_product floats",
         _almost_equal(dot_product([0.5, 0.5], [0.5, 0.5]), 0.5))
    _run("dot_product single element",
         _almost_equal(dot_product([3], [4]), 12))

    _expect_error(dot_product, [[1, 2], [1, 2, 3]], ValueError, "dot_product length mismatch")
    _expect_error(dot_product, ["bad", [1, 2]], TypeError, "dot_product wrong type v1")
    _expect_error(dot_product, [[1, 2], "bad"], TypeError, "dot_product wrong type v2")


# ─────────────────────────────────────────────
# hadamard_product
# ─────────────────────────────────────────────

def test_hadamard_product():
    print("\n--- hadamard_product ---")

    _run("hadamard_product basic",
         hadamard_product([1, 2, 3], [4, 5, 6]) == [4, 10, 18])
    _run("hadamard_product with zeros",
         hadamard_product([1, 2, 3], [0, 0, 0]) == [0, 0, 0])
    _run("hadamard_product negatives",
         hadamard_product([-1, 2], [3, -4]) == [-3, -8])
    _run("hadamard_product floats",
         _list_almost_equal(hadamard_product([0.5, 2.0], [2.0, 0.5]), [1.0, 1.0]))
    _run("hadamard_product single element",
         hadamard_product([6], [7]) == [42])

    _expect_error(hadamard_product, [[1, 2], [1, 2, 3]], ValueError,
                  "hadamard_product length mismatch")
    _expect_error(hadamard_product, ["bad", [1, 2]], TypeError,
                  "hadamard_product wrong type v1")
    _expect_error(hadamard_product, [[1, 2], "bad"], TypeError,
                  "hadamard_product wrong type v2")


# ─────────────────────────────────────────────
# vector_sum
# ─────────────────────────────────────────────

def test_vector_sum():
    print("\n--- vector_sum ---")

    _run("vector_sum integers", _almost_equal(vector_sum([1, 2, 3, 4, 5]), 15))
    _run("vector_sum floats", _almost_equal(vector_sum([0.1, 0.2, 0.3]), 0.6))
    _run("vector_sum negatives", _almost_equal(vector_sum([-1, -2, 3]), 0))
    _run("vector_sum single element", _almost_equal(vector_sum([42]), 42))
    _run("vector_sum empty", _almost_equal(vector_sum([]), 0))

    _expect_error(vector_sum, ["not_a_list"], TypeError, "vector_sum wrong type")


# ─────────────────────────────────────────────
# vector_mean
# ─────────────────────────────────────────────

def test_vector_mean():
    print("\n--- vector_mean ---")

    _run("vector_mean integers", _almost_equal(vector_mean([1, 2, 3, 4, 5]), 3.0))
    _run("vector_mean floats", _almost_equal(vector_mean([1.0, 2.0, 3.0]), 2.0))
    _run("vector_mean single element", _almost_equal(vector_mean([7]), 7.0))
    _run("vector_mean negatives", _almost_equal(vector_mean([-2, 0, 2]), 0.0))

    _expect_error(vector_mean, [[]], ValueError, "vector_mean empty vector")
    _expect_error(vector_mean, ["not_a_list"], TypeError, "vector_mean wrong type")


# ─────────────────────────────────────────────
# argmax
# ─────────────────────────────────────────────

def test_argmax():
    print("\n--- argmax ---")

    _run("argmax basic", argmax([1, 5, 3, 2]) == 1)
    _run("argmax first element max", argmax([9, 1, 2]) == 0)
    _run("argmax last element max", argmax([1, 2, 9]) == 2)
    _run("argmax single element", argmax([42]) == 0)
    _run("argmax negative values", argmax([-5, -1, -3]) == 1)
    _run("argmax tie returns first", argmax([3, 3, 3]) == 0)
    _run("argmax floats", argmax([0.1, 0.9, 0.5]) == 1)

    _expect_error(argmax, [[]], ValueError, "argmax empty vector")
    _expect_error(argmax, ["not_a_list"], TypeError, "argmax wrong type")


# ─────────────────────────────────────────────
# matrix_vector_multiply
# ─────────────────────────────────────────────

def test_matrix_vector_multiply():
    print("\n--- matrix_vector_multiply ---")

    M = [[1, 2, 3],
         [4, 5, 6]]
    v = [1, 1, 1]
    expected = [6, 15]
    _run("matrix_vector_multiply basic",
         _list_almost_equal(matrix_vector_multiply(M, v), expected))

    # identity matrix
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    v2 = [3, 7, 2]
    _run("matrix_vector_multiply identity",
         _list_almost_equal(matrix_vector_multiply(I, v2), v2))

    # 1×1 matrix
    _run("matrix_vector_multiply 1x1",
         _list_almost_equal(matrix_vector_multiply([[5]], [3]), [15]))

    # compare with numpy
    A_np = np.array([[2, -1], [1, 3], [0, 4]])
    v_np = np.array([2, 1])
    expected_np = (A_np @ v_np).tolist()
    A_list = A_np.tolist()
    v_list = v_np.tolist()
    _run("matrix_vector_multiply vs numpy",
         _list_almost_equal(matrix_vector_multiply(A_list, v_list), expected_np))

    # shape mismatches
    _expect_error(matrix_vector_multiply, [[[1, 2, 3], [4, 5, 6]], [1, 2]],
                  ValueError, "matrix_vector_multiply col/vec mismatch")
    _expect_error(matrix_vector_multiply, [[[1, 2]], "bad"],
                  TypeError, "matrix_vector_multiply bad vector type")
    _expect_error(matrix_vector_multiply, ["bad", [1, 2]],
                  TypeError, "matrix_vector_multiply bad matrix type")


# ─────────────────────────────────────────────
# outer_product
# ─────────────────────────────────────────────

def test_outer_product():
    print("\n--- outer_product ---")

    v1 = [1, 2, 3]
    v2 = [4, 5]
    result = outer_product(v1, v2)
    expected = [[4, 5], [8, 10], [12, 15]]
    _run("outer_product basic", _matrix_almost_equal(result, expected))

    # single elements
    _run("outer_product 1x1",
         _matrix_almost_equal(outer_product([3], [4]), [[12]]))

    # compare with numpy
    a_np = np.array([1, 2, 3, 4])
    b_np = np.array([5, 6, 7])
    expected_np = np.outer(a_np, b_np).tolist()
    _run("outer_product vs numpy",
         _matrix_almost_equal(outer_product(a_np.tolist(), b_np.tolist()), expected_np))

    _expect_error(outer_product, [[], [1, 2]], ValueError, "outer_product empty v1")
    _expect_error(outer_product, [[1, 2], []], ValueError, "outer_product empty v2")
    _expect_error(outer_product, ["bad", [1, 2]], TypeError, "outer_product wrong type v1")
    _expect_error(outer_product, [[1, 2], "bad"], TypeError, "outer_product wrong type v2")


# ─────────────────────────────────────────────
# matrix_transpose
# ─────────────────────────────────────────────

def test_matrix_transpose():
    print("\n--- matrix_transpose ---")

    M = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 4], [2, 5], [3, 6]]
    _run("matrix_transpose 2x3 -> 3x2",
         _matrix_almost_equal(matrix_transpose(M), expected))

    # square matrix
    S = [[1, 2], [3, 4]]
    expected_s = [[1, 3], [2, 4]]
    _run("matrix_transpose 2x2",
         _matrix_almost_equal(matrix_transpose(S), expected_s))

    # 1×1
    _run("matrix_transpose 1x1",
         _matrix_almost_equal(matrix_transpose([[7]]), [[7]]))

    # double transpose is identity
    original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _run("matrix_transpose double transpose == original",
         _matrix_almost_equal(matrix_transpose(matrix_transpose(original)), original))

    # compare with numpy
    A_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_np = A_np.T.tolist()
    _run("matrix_transpose vs numpy",
         _matrix_almost_equal(matrix_transpose(A_np.tolist()), expected_np))

    _expect_error(matrix_transpose, ["bad"], TypeError, "matrix_transpose wrong type")
    _expect_error(matrix_transpose, [[]], ValueError, "matrix_transpose zero columns")


# ─────────────────────────────────────────────
# matrix_add
# ─────────────────────────────────────────────

def test_matrix_add():
    print("\n--- matrix_add ---")

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[6, 8], [10, 12]]
    _run("matrix_add basic", _matrix_almost_equal(matrix_add(A, B), expected))

    # add zero matrix
    Z = [[0, 0], [0, 0]]
    _run("matrix_add with zeros", _matrix_almost_equal(matrix_add(A, Z), A))

    # float matrices
    Af = [[0.1, 0.2], [0.3, 0.4]]
    Bf = [[0.9, 0.8], [0.7, 0.6]]
    expected_f = [[1.0, 1.0], [1.0, 1.0]]
    _run("matrix_add floats",
         _matrix_almost_equal(matrix_add(Af, Bf), expected_f))

    # 1×1
    _run("matrix_add 1x1",
         _matrix_almost_equal(matrix_add([[3]], [[4]]), [[7]]))

    # compare with numpy
    A_np = np.array([[1, 2, 3], [4, 5, 6]])
    B_np = np.array([[7, 8, 9], [10, 11, 12]])
    expected_np = (A_np + B_np).tolist()
    _run("matrix_add vs numpy",
         _matrix_almost_equal(matrix_add(A_np.tolist(), B_np.tolist()), expected_np))

    # shape mismatches
    _expect_error(matrix_add, [[[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]],
                  ValueError, "matrix_add column mismatch")
    _expect_error(matrix_add, [[[1, 2]], [[1, 2], [3, 4]]],
                  ValueError, "matrix_add row mismatch")
    _expect_error(matrix_add, ["bad", [[1, 2]]], TypeError,
                  "matrix_add wrong type A")
    _expect_error(matrix_add, [[[1, 2]], "bad"], TypeError,
                  "matrix_add wrong type B")


# ─────────────────────────────────────────────
# scalar_matrix_multiply
# ─────────────────────────────────────────────

def test_scalar_matrix_multiply():
    print("\n--- scalar_matrix_multiply ---")

    M = [[1, 2], [3, 4]]
    _run("scalar_matrix_multiply by 2",
         _matrix_almost_equal(scalar_matrix_multiply(2, M), [[2, 4], [6, 8]]))
    _run("scalar_matrix_multiply by 0",
         _matrix_almost_equal(scalar_matrix_multiply(0, M), [[0, 0], [0, 0]]))
    _run("scalar_matrix_multiply by -1",
         _matrix_almost_equal(scalar_matrix_multiply(-1, M), [[-1, -2], [-3, -4]]))
    _run("scalar_matrix_multiply float scalar",
         _matrix_almost_equal(scalar_matrix_multiply(0.5, [[2.0, 4.0]]), [[1.0, 2.0]]))
    _run("scalar_matrix_multiply 1x1",
         _matrix_almost_equal(scalar_matrix_multiply(3, [[5]]), [[15]]))

    # compare with numpy
    A_np = np.array([[1, 2, 3], [4, 5, 6]])
    expected_np = (3 * A_np).tolist()
    _run("scalar_matrix_multiply vs numpy",
         _matrix_almost_equal(scalar_matrix_multiply(3, A_np.tolist()), expected_np))

    _expect_error(scalar_matrix_multiply, ["bad", [[1, 2]]], TypeError,
                  "scalar_matrix_multiply non-numeric scalar")
    _expect_error(scalar_matrix_multiply, [2, "bad"], TypeError,
                  "scalar_matrix_multiply non-list matrix")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  matrix_ops.py – full test suite")
    print("=" * 55)

    test_vector_add()
    test_vector_subtract()
    test_scalar_vector_multiply()
    test_dot_product()
    test_hadamard_product()
    test_vector_sum()
    test_vector_mean()
    test_argmax()
    test_matrix_vector_multiply()
    test_outer_product()
    test_matrix_transpose()
    test_matrix_add()
    test_scalar_matrix_multiply()

    print("\n" + "=" * 55)
    if _failures:
        print(f"  {len(_failures)} test(s) FAILED:")
        for f in _failures:
            print(f"    ✗  {f}")
    else:
        print("  All tests passed.")
    print("=" * 55)


if __name__ == "__main__":
    run_all_tests()