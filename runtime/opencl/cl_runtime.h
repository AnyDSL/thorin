#ifndef __OPENCL_RT_HPP__
#define __OPENCL_RT_HPP__

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

extern "C"
{

// runtime forward declarations
cl_mem malloc_buffer(void *host);
void free_buffer(cl_mem mem);

void write_buffer(cl_mem dev, void *host);
void read_buffer(cl_mem dev, void *host);

void build_program_and_kernel(const char *file_name, const char *kernel_name, bool);

void set_kernel_arg(void *host, size_t size);
void set_problem_size(size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(const char *kernel_name);
void synchronize();

// SPIR wrappers
cl_mem spir_malloc_buffer(void *host);
void spir_free_buffer(cl_mem mem);

void spir_write_buffer(cl_mem dev, void *host);
void spir_read_buffer(cl_mem dev, void *host);

void spir_build_program_and_kernel_from_binary(const char *file_name, const char *kernel_name);
void spir_build_program_and_kernel_from_source(const char *file_name, const char *kernel_name);

void spir_set_kernel_arg(void *host, size_t size);
void spir_set_problem_size(size_t size_x, size_t size_y, size_t size_z);
void spir_set_config_size(size_t size_x, size_t size_y, size_t size_z);

void spir_launch_kernel(const char *kernel_name);
void spir_synchronize();

// helper functions
void *array(size_t elem_size, size_t width, size_t height);
float random_val(int max);
extern int main_impala();

}

#endif  // __OPENCL_RT_HPP__

