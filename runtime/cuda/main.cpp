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
    dev = malloc_memory(num);
    write_memory(dev, host, num);

    load_kernel("simple-gpu64.nvvm", "simple");
    get_tex_ref("texture");
    bind_tex(dev, CU_AD_FORMAT_SIGNED_INT32);
    set_kernel_arg(&dev);
    set_problem_size(1024, 1, 1);
    set_config_size(128, 1, 1);
    launch_kernel("simple");
    synchronize(); // optional
    read_memory(dev, host, num);
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

    for (unsigned int i=0; i<num; ++i) {
        host[i] = i;
    }


    // CODE TO BE GENERATED: BEGIN
    CUdeviceptr tex;
    tex = malloc_memory(num);
    write_memory(tex, host, num);

    CUdeviceptr out;
    out = malloc_memory(num);
    for (unsigned int i=0; i<num; ++i) {
        host[i] = 0;
    }
    write_memory(out, host, num);

    load_kernel("simple-gpu64.nvvm", "simple_tex");
    get_tex_ref("texture");
    bind_tex(tex, CU_AD_FORMAT_SIGNED_INT32);
    set_kernel_arg(&out);
    set_problem_size(1024, 1, 1);
    set_config_size(128, 1, 1);
    launch_kernel("simple");
    synchronize(); // optional
    read_memory(out, host, num);
    free_memory(out);
    free_memory(tex);
    // CODE TO BE GENERATED: END

    // check result
    for (unsigned int i=0; i<num; ++i) {
        if (host[i] != i) {
            printf("Texture test failed!\n");
            return EXIT_FAILURE;
        }
    }
    printf("Texture test passed!\n");

    return EXIT_SUCCESS;
}

