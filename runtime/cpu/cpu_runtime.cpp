#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>

#include "cpu_runtime.h"
#include "thorin_runtime.h"

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

void parallel_for(int num_threads, int lower, int upper, void *args, uint64_t arg_size, void *fun) {
    // C++11 threads version
    void (*fun_ptr) (void*) = reinterpret_cast<void (*) (void*)>(fun);
    const int count = upper - lower;
    const int linear = count / num_threads;
    const int residual = linear + count % num_threads;

    // Create a pool of threads to execute the task
    std::vector<std::thread> pool(num_threads);

    for (int i = 0; i < num_threads - 1; i++) {
        pool[i] = std::thread([args, fun_ptr, linear]() {
            for (int j = 0; j < linear; j++)
                fun_ptr(args);
        });
    }

    pool[num_threads - 1] = std::thread([args, fun_ptr, residual]() {
        for (int j = 0; j < residual; j++)
            fun_ptr(args);
    });

    // Wait for all the threads to finish
    for (int i = 0; i < num_threads; i++)
        pool[i].join();
}
