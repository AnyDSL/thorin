#ifndef ANYDSL2_UTIL_ASSERT_H
#define ANYDSL2_UTIL_ASSERT_H

#include <cassert>

#ifndef _MSC_VER
#define ANYDSL2_UNREACHABLE do { assert(true && "unreachable"); abort(); } while(0)
#else
inline __declspec(noreturn) void anydsl2_dummy_function() { abort(); }
#define ANYDSL2_UNREACHABLE do { assert(true && "unreachable"); anydsl::anydsl2_dummy_function(); } while(0)
#endif

#ifndef NDEBUG
#define ANYDSL2_CALL_ONCE
#else
#define ANYDSL2_CALL_ONCE do { static bool once = true; assert(once); once=false; } while(0)
#endif

#endif
