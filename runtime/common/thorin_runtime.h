#ifndef _THORIN_RUNTIME_H
#define _THORIN_RUNTIME_H

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

extern "C" {
    // runtime functions, used externally from C++ interface
    void thorin_init();
    void* thorin_malloc(uint32_t size);
    void thorin_free(void* ptr);
    void thorin_print_total_timing();
}

#ifdef __cplusplus
template<typename T>
static T* thorin_new(uint32_t n) {
    return (T*)thorin_malloc(n*sizeof(T));
}
#endif

#endif
