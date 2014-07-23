#ifndef __CUDA_RT_HPP__
#define __CUDA_RT_HPP__

#include <cuda.h>
#include <nvvm.h>

#include "thorin_runtime.h"

extern "C"
{

typedef size_t mem_id;

// NVVM wrappers
mem_id nvvm_malloc_memory(size_t dev, void *host);
void nvvm_free_memory(size_t dev, mem_id mem);

void nvvm_write_memory(size_t dev, mem_id mem, void *host);
void nvvm_read_memory(size_t dev, mem_id mem, void *host);

void nvvm_load_nvvm_kernel(size_t dev, const char *file_name, const char *kernel_name);
void nvvm_load_cuda_kernel(size_t dev, const char *file_name, const char *kernel_name);

void nvvm_set_kernel_arg(size_t dev, void *param);
void nvvm_set_kernel_arg_map(size_t dev, mem_id mem);
void nvvm_set_kernel_arg_tex(size_t dev, mem_id mem, const char *name, CUarray_format format);
void nvvm_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void nvvm_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void nvvm_launch_kernel(size_t dev, const char *kernel_name);
void nvvm_synchronize(size_t dev);

// runtime functions
mem_id map_memory(size_t dev, size_t type_, void *from, int offset, int size);
void unmap_memory(size_t dev, size_t type_, mem_id mem);

}

#endif  // __CUDA_RT_HPP__

