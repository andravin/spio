"""Utility functions for testing matrix operations."""

import torch


def make_test_matrices(
    m: int = 16,
    n: int = 8,
    k: int = 8,
    ones: bool = False,
    a_ones: bool = False,
    b_ones: bool = False,
    output_dtype: torch.dtype = torch.float32,
):
    """Return three matrices of size m x k, k x n, and m x n."""
    if ones or a_ones:
        A = _make_ones_test_matrix(m=m, n=k)
    else:
        A = _make_test_matrix(m=m, n=k, modulus=17)
    if ones or b_ones:
        B = _make_ones_test_matrix(m=k, n=n)
    else:
        B = _make_test_matrix(m=k, n=n, modulus=19)
    C = _make_output_matrix(m=m, n=n, dtype=output_dtype)
    B_trans = torch.transpose(B, 0, 1).contiguous()
    return A, B_trans, C


def _make_test_matrix(
    m: int = 16,
    n: int = 16,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    modulus: int = 17,
):
    matrix = torch.zeros((m, n), dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            matrix[i, j] = (i * n + j) % modulus
    return matrix


def _make_ones_test_matrix(
    m: int = 16,
    n: int = 16,
    dtype: torch.dtype = torch.float16,
    device="cuda",
):
    return torch.ones((m, n), dtype=dtype, device=device)


def matmul_trans_ref(A, B):
    """Return the matrix product A @ B^T."""
    return torch.matmul(A.float(), B.transpose(0, 1).float())


def _make_output_matrix(
    m: int = 16, n: int = 16, dtype: torch.dtype = torch.float32, device="cuda"
):
    return torch.zeros((m, n), dtype=dtype, device=device)
