#ifndef _THORIN_INT_RUNTIME_H
#define _THORIN_INT_RUNTIME_H

extern "C" {
    // runtime functions provided to Impala
    float thorin_random_val();
    void thorin_random_seed(unsigned);
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
} 

#endif
