#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>

#include "cpu_runtime.h"
#include "thorin_runtime.h"

#ifdef USE_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
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
// C++11 threads version
void parallel_for(int num_threads, int lower, int upper, void *args, void *fun) {
    // Get number of available hardware threads
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        // hardware_concurrency is implementation defined, may return 0
        num_threads = (num_threads == 0) ? 1 : num_threads;
    }

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
// TBB version
void parallel_for(int num_threads, int lower, int upper, void *args, void *fun) {
    tbb::task_scheduler_init init((num_threads == 0) ? tbb::task_scheduler_init::automatic : num_threads);
    void (*fun_ptr) (void*, int, int) = reinterpret_cast<void (*) (void*, int, int)>(fun);

    tbb::parallel_for(tbb::blocked_range<int>(lower, upper), [=] (const tbb::blocked_range<int>& range) {
        fun_ptr(args, range.begin(), range.end());
    });
}
#endif

