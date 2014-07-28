#ifndef _THORIN_RUNTIME_H
#define _THORIN_RUNTIME_H

#ifdef __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>
#endif

extern "C" {
    // runtime functions, used externally from C++ interface
    void thorin_init();
    void *thorin_malloc(size_t size);
    void thorin_free(void *ptr);
    void thorin_print_total_timing();
}

#ifdef __cplusplus
template<typename T>
static T* thorin_new(unsigned n)
{
    return (T*)thorin_malloc(n*sizeof(T));
}
#endif

#endif
