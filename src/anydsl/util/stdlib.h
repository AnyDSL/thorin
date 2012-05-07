#include <cstdlib>

#ifdef WIN32
#include <Windows.h>
#include <wchar.h>
#define strtoll _strtoi64
#define strtoull(a,b,c) _strtoui64_l(a, b, c, 0)
#define strtof (float)strtod

#undef TRUE
#undef FALSE
#undef ERROR
#endif
