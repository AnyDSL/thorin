#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cu_runtime.h"

static int num = 1024;

int main_impala() {
    int *host = (int *)malloc(num*sizeof(int));

    for (unsigned int i=0; i<num; ++i) {
        host[i] = 0;
    }

    // CODE TO BE GENERATED: BEGIN
    CUdeviceptr dev;
    dev = malloc_memory(num*sizeof(int));
    write_memory(dev, host, num*sizeof(int));

    load_kernel("simple-gpu64.nvvm", "simple");
    set_kernel_arg(&dev);
    set_problem_size(1024, 1, 1);
    launch_kernel("simple");
    synchronize(); // optional
    read_memory(dev, host, num*sizeof(int));
    free_memory(dev);
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

