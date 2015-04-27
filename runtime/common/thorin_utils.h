#ifndef _THORIN_INT_RUNTIME_H
#define _THORIN_INT_RUNTIME_H

extern "C" {
    // runtime functions provided to Impala
    float thorin_random_val(int);
    long long thorin_get_micro_time();
    void thorin_print_micro_time(long long);
    void thorin_print_gflops(float);
    void* thorin_aligned_malloc(size_t, size_t);
    void thorin_aligned_free(void*);
} 

#endif
