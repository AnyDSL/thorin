#ifndef THORIN_RUNTIME_H
#define THORIN_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum thorin_platform {
    THORIN_HOST = 0,
    THORIN_CUDA = 1,
    THORIN_OPENCL = 2
};

void thorin_info(void);

void* thorin_alloc(int32_t, int32_t, int64_t);
void thorin_release(void*);

void* thorin_map(void*, int64_t, int64_t);
void thorin_unmap(void*);

void thorin_copy(const void*, void*);

void thorin_set_block_size(int32_t, int32_t, int32_t, int32_t, int32_t);
void thorin_set_grid_size(int32_t, int32_t, int32_t, int32_t, int32_t);
void thorin_set_kernel_arg(int32_t, int32_t, int32_t, void*, int32_t);
void thorin_load_kernel(int32_t, int32_t, const char*, const char*);
void thorin_launch_kernel(int32_t, int32_t);
void thorin_synchronize(int32_t, int32_t);

float thorin_random_val();
void thorin_random_seed(int32_t);

long long thorin_get_micro_time();
long long thorin_get_kernel_time();

void thorin_print_char(char);
void thorin_print_int(int);
void thorin_print_long(long long);
void thorin_print_float(float);
void thorin_print_double(double);
void thorin_print_string(char*);

void* thorin_aligned_malloc(size_t, size_t);
void thorin_aligned_free(void*);

#ifdef __cplusplus
}
#endif

#endif
