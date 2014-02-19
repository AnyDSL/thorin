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
cl_mem malloc_buffer(size_t size);
void free_buffer(cl_mem mem);

void write_buffer(cl_mem dev, void *host, size_t size);
void read_buffer(cl_mem dev, void *host, size_t size);

void build_program_and_kernel(const char *file_name, const char *kernel_name, bool);

void set_kernel_arg(void *host, size_t size);
void set_problem_size(size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(const char *kernel_name);
void synchronize();

// SPIR wrappers
inline cl_mem spir_malloc_buffer(size_t size) { return malloc_buffer(size); }
inline void spir_free_buffer(cl_mem mem) { free_buffer(mem); }

inline void spir_write_buffer(cl_mem dev, void *host, size_t size) { write_buffer(dev, host, size); }
inline void spir_read_buffer(cl_mem dev, void *host, size_t size) { read_buffer(dev, host, size); }

inline void spir_build_program_and_kernel_from_binary(const char *file_name, const char *kernel_name) { build_program_and_kernel(file_name, kernel_name, true); }
inline void spir_build_program_and_kernel_from_source(const char *file_name, const char *kernel_name) { build_program_and_kernel(file_name, kernel_name, false); }

inline void spir_set_kernel_arg(void *host, size_t size) { set_kernel_arg(host, size); }
inline void spir_set_problem_size(size_t size_x, size_t size_y, size_t size_z) { set_problem_size(size_x, size_y, size_z); }
inline void spir_set_config_size(size_t size_x, size_t size_y, size_t size_z) { set_config_size(size_x, size_y, size_z); }

inline void spir_launch_kernel(const char *kernel_name) { launch_kernel(kernel_name); }
inline void spir_synchronize() { synchronize(); }

extern int main_impala();
inline float *array(size_t num_elems) {
    return (float *)malloc(sizeof(float)*num_elems);
}
inline float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

}

#endif  // __OPENCL_RT_HPP__

