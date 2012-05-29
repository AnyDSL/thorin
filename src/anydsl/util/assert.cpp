#include <cassert>
#include <cstdlib>

#ifdef WIN32
#include <windows.h>

namespace anydsl {
namespace detail {
void win32_printDebugString(const char* errorMsg) {
    OutputDebugStringA(errorMsg);
}
}
}
#endif

#ifdef __GLIBC__
/*
From:
http://tombarta.wordpress.com/2008/08/01/c-stack-traces-with-gcc/
*/
#include <execinfo.h>
#include <cxxabi.h>
#include <stdio.h>
#include <cstring>
#include <unistd.h>
#endif

namespace anydsl {
namespace detail {
void glibc_printBacktrace() {
#ifdef __GLIBC__
    const size_t stackSize = 16;
    void* stackAddr[stackSize];
    char** stackStrings;
    size_t stackDepth;
    stackDepth = backtrace(stackAddr,stackSize);
    stackStrings = backtrace_symbols(stackAddr, stackDepth);

    for (size_t i = 1; i < stackDepth; i++) {
        size_t sz = 400; // just a guess, template names will go much wider
        char *function = static_cast<char*>(malloc(sz));
        char *begin = 0, *end = 0;
        // find the parentheses and address offset surrounding the mangled name
        for (char *j = stackStrings[i]; *j; ++j) {
            if (*j == '(') {
                begin = j;
            } else if (*j == '+') {
                end = j;
            }
        }
        if (begin && end) {
            *begin++ = '\0';
            *end = '\0';
            // found our mangled name, now in [begin, end)

            int status;
            char *ret = abi::__cxa_demangle(begin, function, &sz, &status);
            if (ret) {
                // return value may be a realloc() of the input
                function = ret;
            } else {
                // demangling failed, just pretend it's a C function with no args
                std::strncpy(function, begin, sz);
                std::strncat(function, "()", sz);
                function[sz-1] = '\0';
            }
            fprintf(stderr, "    %s:%s\n", stackStrings[i], function);
        } else {
            // didn't find the mangled name, just print the whole line
            fprintf(stderr, "    %s\n", stackStrings[i]);
        }
        free(function);
    }

    free(stackStrings);
#endif
}

}
}
