#ifndef _THORIN_INT_RUNTIME_H
#define _THORIN_INT_RUNTIME_H

extern "C" {
    // runtime functions, used internally (by Impala etc.) [TODO: better namespacing]
    float random_val(int);
    long get_micro_time();
    void print_gflops(float);
} 

#endif
