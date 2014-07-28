#ifndef _THORIN_INT_RUNTIME_H
#define _THORIN_INT_RUNTIME_H

extern "C" {
    // runtime functions orovided to Impala
    float thorin_random_val(int);
    long long thorin_get_micro_time();
    void thorin_print_micro_time(long long);
    void thorin_print_gflops(float);
} 

#endif
