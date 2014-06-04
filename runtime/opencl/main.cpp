#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cl_runtime.h"

static int num = 1024;

int main_impala() {
    size_t dev = 1;
    int *host = (int *)thorin_malloc(sizeof(int) * num);

    for (unsigned int i=0; i<num; ++i) {
        host[i] = 0;
    }

    // CODE TO BE GENERATED: BEGIN
    spir_build_program_and_kernel_from_binary(dev, "main.spir.bc", "simple");
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
    for (unsigned int i=0; i<num; ++i) {
        if (host[i] != i) {
            printf("Test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Test passed!\n");

    thorin_free(host);

    return EXIT_SUCCESS;
}

