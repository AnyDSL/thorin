#ifndef __CUDA_RT_HPP__
#define __CUDA_RT_HPP__

#include <cuda.h>
#include <nvvm.h>

extern "C"
{

// runtime forward declarations
CUdeviceptr malloc_memory(size_t size);
void free_memory(CUdeviceptr mem);

void write_memory(CUdeviceptr dev, void *host, size_t size);
void read_memory(CUdeviceptr dev, void *host, size_t size);

void load_kernel(const char *file_name, const char *kernel_name);

void get_tex_ref(const char *name);
void bind_tex(CUdeviceptr mem, CUarray_format format);

void set_kernel_arg(void *host);
void set_problem_size(size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(const char *kernel_name);
void synchronize();

// NVVM wrappers
CUdeviceptr nvvm_malloc_memory(size_t size);
void nvvm_free_memory(CUdeviceptr mem);

void nvvm_write_memory(CUdeviceptr dev, void *host, size_t size);
void nvvm_read_memory(CUdeviceptr dev, void *host, size_t size);

void nvvm_load_kernel(const char *file_name, const char *kernel_name);

void nvvm_get_tex_ref(const char *name);
void nvvm_bind_tex(CUdeviceptr mem, CUarray_format format);

void nvvm_set_kernel_arg(void *host);
void nvvm_set_problem_size(size_t size_x, size_t size_y, size_t size_z);
void nvvm_set_config_size(size_t size_x, size_t size_y, size_t size_z);

void nvvm_launch_kernel(const char *kernel_name);
void nvvm_synchronize();

// helper functions
float *array(size_t num_elems);
float random_val(int max);
extern int main_impala();

}

#endif  // __CUDA_RT_HPP__

