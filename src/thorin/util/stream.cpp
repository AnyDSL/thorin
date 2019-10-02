#include "thorin/util/stream.h"

namespace thorin {

Stream& Stream::endl() {
    ostream() << '\n';
    for (size_t i = 0; i != level_; ++i) ostream() << tab_;
    return *this;
}

Stream& Stream::fmt(const char* s) {
    while (*s) {
        auto next = s + 1;

        if (false) {
        } else if (*s == '\n') { s++; endl();
        } else if (*s == '\t') { s++; indent();
        } else if (*s == '\b') { s++; dedent();
        } else if (*s == '{') {
            if (match2nd(next, s, '{')) continue;

            while (*s && *s != '}') s++;

            assert(*s != '}' && "invalid format string for 'streamf': missing argument(s)");
            assert(false && "invalid format string for 'streamf': missing closing brace and argument");

        } else if (*s == '}') {
            if (match2nd(next, s, '}')) continue;
            assert(false && "unmatched/unescaped closing brace '}' in format string");
        } else {
            (*this) << *s++;
        }
    }
    return *this;
}

bool Stream::match2nd(const char* next, const char*& s, const char c) {
    if (*next == c) {
        (*this) << c;
        s += 2;
        return true;
    }
    return false;
}

}
