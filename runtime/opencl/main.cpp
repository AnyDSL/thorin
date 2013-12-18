#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cl_runtime.h"

static int num = 1024;

int main_impala() {
    int *host = (int *)malloc(num*sizeof(int));

    for (unsigned int i=0; i<num; ++i) {
        host[i] = 0;
    }

    // CODE TO BE GENERATED: BEGIN
    init_opencl(CL_DEVICE_TYPE_CPU);
    build_program_and_kernel("simple-gpu64.spir.bc", "simple");
    cl_mem dev;
    dev = malloc_buffer(num*sizeof(int));
    write_buffer(dev, host, num);

    set_problem_size(1024, 1, 1);
    set_kernel_arg(&dev, sizeof(dev));
    launch_kernel("simple");
    synchronize(); // optional
    read_buffer(dev, host, num);
    free_buffer(dev);
    // CODE TO BE GENERATED: END

    // check result
    for (unsigned int i=0; i<num; ++i) {
        if (host[i] != i) {
            printf("Test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Test passed!\n");

    return EXIT_SUCCESS;
}

