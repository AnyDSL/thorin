#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cl_runtime.h"

static int num = 1024;

int test_kernelfile(const char *file) {
    printf("Test file: %s\n", file);

    size_t dev = 0;
    int *cmem = (int *)thorin_malloc(sizeof(int) * 32);
    int *host = (int *)thorin_malloc(sizeof(int) * num);

    // CODE TO BE GENERATED: BEGIN
    for (size_t i=0; i<num; ++i) host[i] = 0;
    spir_build_program_and_kernel_from_binary(dev, file, "simple");
    mem_id mem = spir_malloc_buffer(dev, host);
    spir_write_buffer(dev, mem, host);
    spir_set_problem_size(dev, 1024, 1, 1);
    spir_set_config_size(dev, 128, 1, 1);
    spir_set_kernel_arg_map(dev, mem);
    spir_launch_kernel(dev, "simple");
    spir_synchronize(dev); // optional
    spir_read_buffer(dev, mem, host);
    spir_free_buffer(dev, mem);
    // CODE TO BE GENERATED: END

    // check result
    for (size_t i=0; i<num; ++i) {
        if (host[i] != i) {
            printf("Test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Test passed!\n");


    // CODE TO BE GENERATED: BEGIN
    for (size_t i=0; i<32; ++i)  cmem[i] = i;
    for (size_t i=0; i<num; ++i) host[i] = 0;
    spir_build_program_and_kernel_from_binary(dev, file, "simple_cmem");
    mem = spir_malloc_buffer(dev, host);
    spir_write_buffer(dev, mem, host);
    spir_set_kernel_arg_map(dev, mem);
    spir_set_kernel_arg_const(dev, cmem, sizeof(int) * 32);
    spir_set_problem_size(dev, 1024, 1, 1);
    spir_set_config_size(dev, 32, 1, 1);
    spir_launch_kernel(dev, "simple_cmem");
    spir_synchronize(dev); // optional
    spir_read_buffer(dev, mem, host);
    spir_free_buffer(dev, mem);
    // CODE TO BE GENERATED: END

    // check result
    for (size_t i=0; i<num; ++i) {
        if (host[i] != i%32) {
            printf("Constant test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Constant test passed!\n");

    thorin_free(cmem);
    thorin_free(host);

    return EXIT_SUCCESS;
}

extern "C" { int main_impala(void); }
int main_impala() {
    return test_kernelfile("main.spir.bc");
}

