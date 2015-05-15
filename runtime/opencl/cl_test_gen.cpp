#include <cstring>
#include <iostream>

#include "cl_runtime.h"

#include "thorin_runtime.h"

static void spir_load_any_kernel(uint32_t dev, const char* file, const char* kernel) {
    size_t len = strlen(file);
    if (len > 3 && !strcmp(file+len-3, ".cl"))
        spir_build_program_and_kernel_from_source(dev, file, kernel);
    else
        spir_build_program_and_kernel_from_binary(dev, file, kernel);
}

static int num = 1024;

int test_kernelfile(const char* file) {
    std::cout << "Test file: " << file << std::endl;

    uint32_t dev = 0;
    int* cmem = (int*)thorin_malloc(sizeof(int) * 32);
    int* host = (int*)thorin_malloc(sizeof(int) * num);


    // CODE TO BE GENERATED: BEGIN
    for (size_t i=0; i<num; ++i) host[i] = 0;
    mem_id mem = spir_malloc_buffer(dev, host);
    spir_write_buffer(dev, mem, host);
    spir_load_any_kernel(dev, file, "simple");
    spir_set_kernel_arg_map(dev, mem);
    spir_set_problem_size(dev, 1024, 1, 1);
    spir_set_config_size(dev, 128, 1, 1);
    spir_launch_kernel(dev, "simple");
    spir_synchronize(dev); // optional
    spir_read_buffer(dev, mem, host);
    spir_free_buffer(dev, mem);
    // CODE TO BE GENERATED: END

    // check result
    for (size_t i=0; i<num; ++i) {
        if (host[i] != i) {
            std::cout << "Test failed!" << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Test passed!" << std::endl;
    std::cout << std::endl;


    // CODE TO BE GENERATED: BEGIN
    for (size_t i=0; i<32; ++i)  cmem[i] = i;
    for (size_t i=0; i<num; ++i) host[i] = 0;
    mem = spir_malloc_buffer(dev, host);
    spir_write_buffer(dev, mem, host);
    spir_load_any_kernel(dev, file, "simple_cmem");
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
            std::cout << "Constant test failed!" << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Constant test passed!" << std::endl;
    std::cout << std::endl;

    thorin_free(cmem);
    thorin_free(host);

    return EXIT_SUCCESS;
}

extern "C"
int cl_test() {
    int ret = test_kernelfile("cl_test.cl");
    if (ret == EXIT_SUCCESS)
        ret = test_kernelfile("cl_test.spir.bc");
    return ret;
}

