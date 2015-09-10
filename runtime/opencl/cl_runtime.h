#ifndef __OPENCL_RT_HPP__
#define __OPENCL_RT_HPP__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

extern "C"
{

typedef uint32_t mem_id;

// SPIR wrappers
mem_id spir_malloc_buffer(uint32_t dev, void* host);
void spir_free_buffer(uint32_t dev, mem_id mem);

void spir_write_buffer(uint32_t dev, mem_id mem, void* host);
void spir_read_buffer(uint32_t dev, mem_id mem, void* host);

void spir_build_program_and_kernel_from_binary(uint32_t dev, const char* file_name, const char* kernel_name);
void spir_build_program_and_kernel_from_source(uint32_t dev, const char* file_name, const char* kernel_name);

void spir_set_kernel_arg(uint32_t dev, void* param, uint32_t size);
void spir_set_kernel_arg_map(uint32_t dev, mem_id mem);
void spir_set_kernel_arg_const(uint32_t dev, void* param, uint32_t size);
void spir_set_kernel_arg_struct(uint32_t dev, void* param, uint32_t size);
void spir_set_problem_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);
void spir_set_config_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);

void spir_launch_kernel(uint32_t dev, const char* kernel_name);
void spir_synchronize(uint32_t dev);

// runtime functions
mem_id map_memory(uint32_t dev, uint32_t type_, void* from, int offset, int size);
void unmap_memory(mem_id mem);

}

#endif  // __OPENCL_RT_HPP__

