import numpy as np


# ─────────────────────────────────────────────
# Vector operations
# ─────────────────────────────────────────────

def vector_add(v1: list, v2: list) -> list:
    """
    Element-wise addition of two vectors.

    Parameters
    ----------
    v1 : list  First input vector.
    v2 : list  Second input vector.

    Returns
    -------
    list  Element-wise sum.

    Raises
    ------
    ValueError  If v1 and v2 have different lengths.
    TypeError   If either argument is not a list.
    """
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError(
            f"vector_add expects two lists, "
            f"got {type(v1).__name__} and {type(v2).__name__}."
        )
    if len(v1) != len(v2):
        raise ValueError(
            f"Shape mismatch in vector_add: "
            f"v1 has length {len(v1)}, v2 has length {len(v2)}."
        )
    return [a + b for a, b in zip(v1, v2)]


def vector_subtract(v1: list, v2: list) -> list:
    """
    Element-wise subtraction of two vectors (v1 - v2).

    Parameters
    ----------
    v1 : list  First input vector (minuend).
    v2 : list  Second input vector (subtrahend).

    Returns
    -------
    list  Element-wise difference.

    Raises
    ------
    ValueError  If v1 and v2 have different lengths.
    TypeError   If either argument is not a list.
    """
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError(
            f"vector_subtract expects two lists, "
            f"got {type(v1).__name__} and {type(v2).__name__}."
        )
    if len(v1) != len(v2):
        raise ValueError(
            f"Shape mismatch in vector_subtract: "
            f"v1 has length {len(v1)}, v2 has length {len(v2)}."
        )
    return [a - b for a, b in zip(v1, v2)]


def scalar_vector_multiply(scalar: float, v: list) -> list:
    """
    Multiply every element of a vector by a scalar.

    Parameters
    ----------
    scalar : float  The scalar multiplier.
    v      : list   The input vector.

    Returns
    -------
    list  Scaled vector.

    Raises
    ------
    TypeError  If scalar is not a number or v is not a list.
    """
    if not isinstance(scalar, (int, float)):
        raise TypeError(
            f"scalar_vector_multiply expects a numeric scalar, "
            f"got {type(scalar).__name__}."
        )
    if not isinstance(v, list):
        raise TypeError(
            f"scalar_vector_multiply expects a list vector, "
            f"got {type(v).__name__}."
        )
    return [scalar * x for x in v]


def dot_product(v1: list, v2: list) -> float:
    """
    Compute the dot product (inner product) of two vectors.

    Parameters
    ----------
    v1 : list  First input vector.
    v2 : list  Second input vector.

    Returns
    -------
    float  Sum of element-wise products.

    Raises
    ------
    ValueError  If v1 and v2 have different lengths.
    TypeError   If either argument is not a list.
    """
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError(
            f"dot_product expects two lists, "
            f"got {type(v1).__name__} and {type(v2).__name__}."
        )
    if len(v1) != len(v2):
        raise ValueError(
            f"Shape mismatch in dot_product: "
            f"v1 has length {len(v1)}, v2 has length {len(v2)}."
        )
    return sum(a * b for a, b in zip(v1, v2))


def hadamard_product(v1: list, v2: list) -> list:
    """
    Element-wise (Hadamard) product of two vectors.

    Parameters
    ----------
    v1 : list  First input vector.
    v2 : list  Second input vector.

    Returns
    -------
    list  Element-wise products.

    Raises
    ------
    ValueError  If v1 and v2 have different lengths.
    TypeError   If either argument is not a list.
    """
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError(
            f"hadamard_product expects two lists, "
            f"got {type(v1).__name__} and {type(v2).__name__}."
        )
    if len(v1) != len(v2):
        raise ValueError(
            f"Shape mismatch in hadamard_product: "
            f"v1 has length {len(v1)}, v2 has length {len(v2)}."
        )
    return [a * b for a, b in zip(v1, v2)]


def vector_sum(v: list) -> float:
    """
    Return the sum of all elements in a vector.

    Parameters
    ----------
    v : list  Input vector.

    Returns
    -------
    float  Sum of all elements.

    Raises
    ------
    TypeError  If v is not a list.
    """
    if not isinstance(v, list):
        raise TypeError(
            f"vector_sum expects a list, got {type(v).__name__}."
        )
    return sum(v)


def vector_mean(v: list) -> float:
    """
    Return the arithmetic mean of all elements in a vector.

    Parameters
    ----------
    v : list  Input vector.

    Returns
    -------
    float  Mean of all elements.

    Raises
    ------
    TypeError   If v is not a list.
    ValueError  If v is empty (mean of empty sequence is undefined).
    """
    if not isinstance(v, list):
        raise TypeError(
            f"vector_mean expects a list, got {type(v).__name__}."
        )
    if len(v) == 0:
        raise ValueError(
            "vector_mean received an empty vector; mean is undefined."
        )
    return sum(v) / len(v)


def argmax(v: list) -> int:
    """
    Return the index of the largest element in a vector.
    In the case of a tie the first occurrence is returned.

    Parameters
    ----------
    v : list  Input vector.

    Returns
    -------
    int  Index of the maximum element.

    Raises
    ------
    TypeError   If v is not a list.
    ValueError  If v is empty.
    """
    if not isinstance(v, list):
        raise TypeError(
            f"argmax expects a list, got {type(v).__name__}."
        )
    if len(v) == 0:
        raise ValueError(
            "argmax received an empty vector."
        )
    max_idx = 0
    for i in range(1, len(v)):
        if v[i] > v[max_idx]:
            max_idx = i
    return max_idx


# ─────────────────────────────────────────────
# Matrix helpers
# ─────────────────────────────────────────────

def _check_matrix(M: list, name: str = "M") -> tuple:
    """
    Validate that M is a non-empty rectangular list-of-lists and return
    its shape (rows, cols).

    Parameters
    ----------
    M    : list  The candidate matrix (list of lists).
    name : str   Variable name used in error messages.

    Returns
    -------
    tuple  (num_rows, num_cols)

    Raises
    ------
    TypeError   If M is not a list of lists.
    ValueError  If M is empty, has no columns, or is not rectangular.
    """
    if not isinstance(M, list) or not all(isinstance(row, list) for row in M):
        raise TypeError(
            f"{name} must be a list of lists (2-D matrix)."
        )
    if len(M) == 0:
        raise ValueError(f"{name} is an empty matrix.")
    num_cols = len(M[0])
    if num_cols == 0:
        raise ValueError(f"{name} has rows with zero columns.")
    for i, row in enumerate(M):
        if len(row) != num_cols:
            raise ValueError(
                f"{name} is not rectangular: row 0 has {num_cols} columns "
                f"but row {i} has {len(row)} columns."
            )
    return len(M), num_cols


# ─────────────────────────────────────────────
# Matrix operations
# ─────────────────────────────────────────────

def matrix_vector_multiply(M: list, v: list) -> list:
    """
    Multiply a matrix M (m × n) by a vector v of length n,
    returning a vector of length m.

    Each element i of the result is the dot product of row i of M with v.

    Parameters
    ----------
    M : list  2-D list with shape (m, n).
    v : list  1-D list of length n.

    Returns
    -------
    list  Result vector of length m.

    Raises
    ------
    ValueError  If the number of columns in M does not equal len(v).
    TypeError   If M is not a list of lists or v is not a list.
    """
    m, n = _check_matrix(M, "M")
    if not isinstance(v, list):
        raise TypeError(
            f"matrix_vector_multiply expects a list for v, "
            f"got {type(v).__name__}."
        )
    if n != len(v):
        raise ValueError(
            f"Shape mismatch in matrix_vector_multiply: "
            f"M has {n} columns but v has length {len(v)}."
        )
    return [dot_product(row, v) for row in M]


def outer_product(v1: list, v2: list) -> list:
    """
    Compute the outer product of two vectors.

    Given v1 of length m and v2 of length n, return a matrix of shape
    (m × n) where element [i][j] = v1[i] * v2[j].

    Parameters
    ----------
    v1 : list  First vector (length m).
    v2 : list  Second vector (length n).

    Returns
    -------
    list  2-D list of shape (m, n).

    Raises
    ------
    TypeError  If either argument is not a list.
    ValueError If either vector is empty.
    """
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError(
            f"outer_product expects two lists, "
            f"got {type(v1).__name__} and {type(v2).__name__}."
        )
    if len(v1) == 0 or len(v2) == 0:
        raise ValueError(
            "outer_product received an empty vector."
        )
    return [[a * b for b in v2] for a in v1]


def matrix_transpose(M: list) -> list:
    """
    Transpose a matrix (swap rows and columns).

    For a matrix of shape (m × n) the result has shape (n × m).

    Parameters
    ----------
    M : list  2-D list with shape (m, n).

    Returns
    -------
    list  Transposed 2-D list with shape (n, m).

    Raises
    ------
    TypeError   If M is not a list of lists.
    ValueError  If M is empty or not rectangular.
    """
    m, n = _check_matrix(M, "M")
    return [[M[r][c] for r in range(m)] for c in range(n)]


def matrix_add(A: list, B: list) -> list:
    """
    Element-wise addition of two matrices.

    Parameters
    ----------
    A : list  2-D list with shape (m, n).
    B : list  2-D list with shape (m, n).

    Returns
    -------
    list  Element-wise sum, shape (m, n).

    Raises
    ------
    ValueError  If A and B do not have the same shape.
    TypeError   If either argument is not a list of lists.
    """
    m_a, n_a = _check_matrix(A, "A")
    m_b, n_b = _check_matrix(B, "B")
    if m_a != m_b or n_a != n_b:
        raise ValueError(
            f"Shape mismatch in matrix_add: "
            f"A is ({m_a} × {n_a}) but B is ({m_b} × {n_b})."
        )
    return [
        [A[r][c] + B[r][c] for c in range(n_a)]
        for r in range(m_a)
    ]


def scalar_matrix_multiply(scalar: float, M: list) -> list:
    """
    Multiply every element of a matrix by a scalar.

    Parameters
    ----------
    scalar : float  The scalar multiplier.
    M      : list   2-D list with shape (m, n).

    Returns
    -------
    list  Scaled matrix, same shape as M.

    Raises
    ------
    TypeError   If scalar is not numeric or M is not a list of lists.
    ValueError  If M is empty or not rectangular.
    """
    if not isinstance(scalar, (int, float)):
        raise TypeError(
            f"scalar_matrix_multiply expects a numeric scalar, "
            f"got {type(scalar).__name__}."
        )
    m, n = _check_matrix(M, "M")
    return [
        [scalar * M[r][c] for c in range(n)]
        for r in range(m)
    ]