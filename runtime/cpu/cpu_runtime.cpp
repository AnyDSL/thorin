#include <stdlib.h>
#include <iostream>

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

