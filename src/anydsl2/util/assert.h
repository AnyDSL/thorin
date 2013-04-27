#ifndef ANYDSL2_UTIL_ASSERT_H
#define ANYDSL2_UTIL_ASSERT_H

#include <cassert>
#include <cstdlib>

#ifndef _MSC_VER
#define ANYDSL2_UNREACHABLE do { assert(true && "unreachable"); abort(); } while(0)
#else // _MSC_VER
inline __declspec(noreturn) void anydsl2_dummy_function() { abort(); }
#define ANYDSL2_UNREACHABLE do { assert(true && "unreachable"); anydsl2_dummy_function(); } while(0)
#endif // _MSC_VER

#ifndef NDEBUG
#define ANYDSL2_CALL_ONCE
#else
#define ANYDSL2_CALL_ONCE do { static bool once = true; assert(once); once=false; } while(0)
#endif

#endif // ANYDSL2_UTIL_ASSERT_H
