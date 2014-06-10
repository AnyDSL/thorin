#include <stdlib.h>
#include <iostream>
#include <thorin_runtime.h>

// helper functions
void thorin_init() {  }
void *thorin_malloc(size_t size) {
    void *mem;
    posix_memalign(&mem, 64, size);
    std::cerr << " * malloc() -> " << mem << std::endl;
    return mem;
}
void thorin_free(void *ptr) {
    free(ptr);
}


