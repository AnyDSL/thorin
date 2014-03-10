#ifndef __CUDA_RT_HPP__
#define __CUDA_RT_HPP__

#include <cuda.h>
#include <nvvm.h>

extern "C"
{

typedef size_t mem_id;

// runtime forward declarations
mem_id malloc_memory(size_t dev, void *host);
void free_memory(size_t dev, mem_id mem);

void write_memory(size_t dev, mem_id mem, void *host);
void read_memory(size_t dev, mem_id mem, void *host);

void load_kernel(size_t dev, const char *file_name, const char *kernel_name);

void get_tex_ref(size_t dev, const char *name);
void bind_tex(size_t dev, mem_id mem, CUarray_format format);

void set_kernel_arg(size_t dev, void *param);
void set_kernel_arg_map(size_t dev, mem_id mem);
void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(size_t dev, const char *kernel_name);
void synchronize(size_t dev);

// NVVM wrappers
mem_id nvvm_malloc_memory(size_t dev, void *host);
void nvvm_free_memory(size_t dev, mem_id mem);

void nvvm_write_memory(size_t dev, mem_id mem, void *host);
void nvvm_read_memory(size_t dev, mem_id mem, void *host);

void nvvm_load_kernel(size_t dev, const char *file_name, const char *kernel_name);

void nvvm_set_kernel_arg(size_t dev, void *param);
void nvvm_set_kernel_arg_map(size_t dev, mem_id mem);
void nvvm_set_kernel_arg_tex(size_t dev, mem_id mem, char *name, CUarray_format format);
void nvvm_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void nvvm_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void nvvm_launch_kernel(size_t dev, const char *kernel_name);
void nvvm_synchronize(size_t dev);

// helper functions
void *array(size_t elem_size, size_t width, size_t height);
void free_array(void *host);
mem_id map_memory(size_t dev, size_t type, void *from, int ox, int oy, int oz, int sx, int sy, int sz);
void unmap_memory(size_t dev, size_t type_, mem_id mem);
float random_val(int max);
extern int main_impala();

}

#endif  // __CUDA_RT_HPP__

