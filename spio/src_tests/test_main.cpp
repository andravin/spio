#include "utest.h"
#include <iostream>

UTEST_STATE();

// Run a specific test:
// int new_argc = 2;
// const char *const new_argv[] = {"spio_cpp_tests", "--filter=Tensor.tensor_2d_small"};
// return utest_main(new_argc, new_argv);
int main(int argc, const char* const argv[]) {
    return utest_main(argc, argv);
}
