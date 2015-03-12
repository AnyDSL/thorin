#ifndef __CUDA_RT_HPP__
#define __CUDA_RT_HPP__

#include <cuda.h>
#include <nvvm.h>

#include <cstdint>

#include "thorin_runtime.h"

extern "C"
{

typedef uint32_t mem_id;

// NVVM wrappers
mem_id nvvm_malloc_memory(uint32_t dev, void *host);
void nvvm_free_memory(uint32_t dev, mem_id mem);

void nvvm_write_memory(uint32_t dev, mem_id mem, void *host);
void nvvm_read_memory(uint32_t dev, mem_id mem, void *host);

void nvvm_load_nvvm_kernel(uint32_t dev, const char *file_name, const char *kernel_name);
void nvvm_load_cuda_kernel(uint32_t dev, const char *file_name, const char *kernel_name);

void nvvm_set_kernel_arg(uint32_t dev, void *param);
void nvvm_set_kernel_arg_map(uint32_t dev, mem_id mem);
void nvvm_set_kernel_arg_tex(uint32_t dev, mem_id mem, const char *name, CUarray_format format);
void nvvm_set_kernel_arg_const(uint32_t dev, void *param, const char *name, uint32_t size);
void nvvm_set_problem_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);
void nvvm_set_config_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);

void nvvm_launch_kernel(uint32_t dev, const char *kernel_name);
void nvvm_synchronize(uint32_t dev);

// runtime functions
mem_id map_memory(uint32_t dev, uint32_t type_, void *from, int offset, int size);
void unmap_memory(mem_id mem);

}

#endif  // __CUDA_RT_HPP__

