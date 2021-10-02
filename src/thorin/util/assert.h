#ifndef THORIN_UTIL_ASSERT_H
#define THORIN_UTIL_ASSERT_H

#include <cassert>

namespace thorin {
[[noreturn]] inline void _unreachable() { abort(); }
}

#define THORIN_UNREACHABLE do { assert(false && "unreachable"); thorin::_unreachable(); } while(0)

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define THORIN_BREAK asm("int3");
#else
#define THORIN_BREAK { int* __p__ = nullptr; *__p__ = 42; }
#endif

#ifndef NDEBUG
#define assert_unused(x) assert(x)
#else
#define assert_unused(x) ((void) (0 && (x)))
#endif

#endif
