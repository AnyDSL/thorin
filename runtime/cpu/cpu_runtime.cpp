#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>

#include "cpu_runtime.h"
#include "thorin_runtime.h"

#ifdef USE_TBB
#include "tbb/parallel_for.h"
#endif

// helper functions
void thorin_init() { }
void *thorin_malloc(uint32_t size) {
    void *mem;
    posix_memalign(&mem, 64, size);
    std::cerr << " * malloc(" << size << ") -> " << mem << std::endl;
    return mem;
}
void thorin_free(void *ptr) {
    free(ptr);
}
void thorin_print_total_timing() { }

#ifndef USE_TBB
void parallel_for(int num_threads, int lower, int upper, void *args, void *fun) {
    // C++11 threads version
    void (*fun_ptr) (void*, int, int) = reinterpret_cast<void (*) (void*, int, int)>(fun);
    const int linear = (upper - lower) / num_threads;

    // Create a pool of threads to execute the task
    std::vector<std::thread> pool(num_threads);

    for (int i = 0, a = lower, b = lower + linear; i < num_threads - 1; a = b, b += linear, i++) {
        pool[i] = std::thread([=]() {
            fun_ptr(args, a, b);
        });
    }

    pool[num_threads - 1] = std::thread([=]() {
        fun_ptr(args, lower + (num_threads - 1) * linear, upper);
    });

    // Wait for all the threads to finish
    for (int i = 0; i < num_threads; i++)
        pool[i].join();
}
#else
void parallel_for(int num_threads, int lower, int upper, void *args, uint64_t arg_size, void *fun) {
    // TBB version
    void (*fun_ptr) (void*, int, int) = reinterpret_cast<void (*) (void*, int, int)>(fun);

    tbb::parallel_for(lower, upper, [] (const blocked_range<int>& range) {
        fun_ptr(args, range.begin(), range.end());
    });
}
#endif

