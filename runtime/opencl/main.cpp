#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cl_runtime.h"

static int num = 1024;

int main_impala() {
    size_t dev = 0;
    int *host = (int *)array(sizeof(int), num, 1);

    for (unsigned int i=0; i<num; ++i) {
        host[i] = 0;
    }

    // CODE TO BE GENERATED: BEGIN
    bool is_binary = true;
    build_program_and_kernel(dev, "simple-gpu64.spir.bc", "simple", is_binary);
    cl_mem mem;
    mem = malloc_buffer(dev, host);
    write_buffer(dev, mem, host);

    set_problem_size(dev, 1024, 1, 1);
    set_config_size(dev, 128, 1, 1);
    set_kernel_arg(dev, &mem, sizeof(mem));
    launch_kernel(dev, "simple");
    synchronize(dev); // optional
    read_buffer(dev, mem, host);
    free_buffer(dev, mem);
    // CODE TO BE GENERATED: END

    // check result
    for (unsigned int i=0; i<num; ++i) {
        if (host[i] != i) {
            printf("Test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Test passed!\n");

    free(host);

    return EXIT_SUCCESS;
}

