#ifndef DSLU_STRING_HEADER
#define DSLU_STRING_HEADER

#include "std.h"

namespace anydsl {
struct StrCmp {
    bool operator()(const char* s1, const char* s2) const { return strcmp(s1, s2) < 0; }
};
}

#endif

