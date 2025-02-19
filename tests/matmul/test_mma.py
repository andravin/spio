"""Test matrix multiplication using matrix fragments and tensor cores."""

from spio.compiler import compile_and_load_kernel
from spio.util import assert_all_close_with_acc_depth

from spio.util.test_matrices import make_test_matrices, matmul_trans_ref


def test_mma_m16_n8_k8_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k8."""
    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_m16_n8_k8",
        source_file_name="mma.cu",
        src_module="spio.src_tests",
    )

    A, B_trans, C = make_test_matrices(m=16, n=8, k=8)

    mma_kernel.launch((1, 1, 1), (32, 1, 1), (C, A, B_trans))

    C_ref = matmul_trans_ref(A, B_trans)

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=8)


def test_mma_m16_n8_k16_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k16."""
    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_m16_n8_k16",
        source_file_name="mma.cu",
        debug=True,
        src_module="spio.src_tests",
    )

    A, B_trans, C = make_test_matrices(m=16, n=8, k=16)

    mma_kernel.launch((1, 1, 1), (32, 1, 1), (C, A, B_trans))

    C_ref = matmul_trans_ref(A, B_trans)

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=16)


def test_mma_m16_n16_k16_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n16_k16."""
    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_m16_n16_k16",
        source_file_name="mma.cu",
        debug=True,
        src_module="spio.src_tests",
    )

    A, B_trans, C = make_test_matrices(m=16, n=16, k=16)

    mma_kernel.launch((1, 1, 1), (32, 1, 1), (C, A, B_trans))

    C_ref = matmul_trans_ref(A, B_trans)

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=16)
