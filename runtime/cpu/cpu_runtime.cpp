#include <stdlib.h>
#include <iostream>

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

// pthreads / TBB runtime
void parallel_for(int num_threads, int lower, int upper, void *args, uint64_t arg_size, void *fun) {
    // TODO
}

