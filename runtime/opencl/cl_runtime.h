#ifndef __OPENCL_RT_HPP__
#define __OPENCL_RT_HPP__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

extern "C"
{

typedef size_t mem_id;

// runtime forward declarations
mem_id malloc_buffer(size_t dev, void *host, cl_mem_flags flags=CL_MEM_READ_WRITE);
void free_buffer(size_t dev, mem_id mem);

void write_buffer(size_t dev, mem_id mem, void *host);
void read_buffer(size_t dev, mem_id mem, void *host);

void build_program_and_kernel(size_t dev, const char *file_name, const char *kernel_name, bool);

void set_kernel_arg(size_t dev, void *param, size_t size);
void set_kernel_arg_map(size_t dev, mem_id mem);
void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(size_t dev, const char *kernel_name);
void synchronize(size_t dev);

// SPIR wrappers
mem_id spir_malloc_buffer(size_t dev, void *host);
void spir_free_buffer(size_t dev, mem_id mem);

void spir_write_buffer(size_t dev, mem_id mem, void *host);
void spir_read_buffer(size_t dev, mem_id mem, void *host);

void spir_build_program_and_kernel_from_binary(size_t dev, const char *file_name, const char *kernel_name);
void spir_build_program_and_kernel_from_source(size_t dev, const char *file_name, const char *kernel_name);

void spir_set_kernel_arg(size_t dev, void *param, size_t size);
void spir_set_kernel_arg_map(size_t dev, mem_id mem);
void spir_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void spir_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void spir_launch_kernel(size_t dev, const char *kernel_name);
void spir_synchronize(size_t dev);

// helper functions
void *array(size_t elem_size, size_t width, size_t height);
void free_array(void *host);
mem_id map_memory(size_t dev, size_t type, void *from, int ox, int oy, int oz, int sx, int sy, int sz);
void unmap_memory(size_t dev, size_t type_, mem_id mem);
float random_val(int max);
void getMicroTime();
extern int main_impala();

}

#endif  // __OPENCL_RT_HPP__

