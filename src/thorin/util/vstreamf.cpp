#include "log.h"

#include <cstdlib>
#include <cstring>
#include <ios>
#include <iostream>
#include <new>
#include <stdexcept>

#include "vstreamf.h"

namespace thorin {

void Streamable::dump() const { stream(std::cout); }

std::ostream& operator << (std::ostream& ostream, const Streamable* s) { return s->stream(ostream); }

static inline char const* strstart(char const* str, char const* start) {
	do {
		if (*start == '\0')
			return str;
	} while (*str++ == *start++);
	return NULL;
}

static void streamf(std::ostream& out, char const *fmt, ...) {
    char* msg;
    va_list ap;
    va_start(ap, fmt);

#ifdef _GNU_SOURCE
    vasprintf(&msg, fmt, ap);
#else
    // determine required size
    int size = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);

    if (size < 0)
        throw std::ios_base::failure("output error: cannot use vsnprintf");

    size++; // for '\0'
    msg = (char*) malloc(size);
    if (msg == nullptr)
        throw std::bad_alloc();

    va_start(ap, fmt);
    size = vsnprintf(msg, size, fmt, ap);
    if (size < 0) {
        free(msg);
        throw std::ios_base::failure("output error: cannot use vsnprintf");
    }
#endif

    va_end(ap);
    out << msg;
    free(msg);
}

void vstreamf(std::ostream& out, char const *fmt, va_list ap) {
    for (char const *f; (f = strchr(fmt, '%')); fmt = f) {
        out.write(fmt, f++ - fmt); // write till '%' and skip '%'

        bool flag_zero, flag_long = false;
        for (; ; ++f) {
            switch (*f) {
                case '0':
                    flag_zero = true;
                    break;
                case 'l':
                    flag_long = true;
                    break;
                default:
                    goto done_flags;
            }
        }
        done_flags:;

        int field_width = 0;
        if (*f == '*') {
            ++f;
            field_width = va_arg(ap, int);
        }

        int precision = -1;
        if (auto rest = strstart(f, ".*")) {
            f = rest;
            precision = va_arg(ap, int);
        }

        switch (*f++) {
            case '%':
                out << '%';
                break;
            case 'c':
                out.put(va_arg(ap, int));
                break;
            case 'i':
            case 'd':
                if (flag_long)
                    streamf(out, "%ld", va_arg(ap, long));
                else
                    streamf(out, "%d", va_arg(ap, int));
                break;
            case 's':
                streamf(out, "%.*s", precision, va_arg(ap, const char*));
                break;
            case 'S':
                out << va_arg(ap, const std::string);
                break;
            case 'u':
                streamf(out, "%u", va_arg(ap, unsigned int));
                break;
            case 'X': {
                auto val = va_arg(ap, unsigned int);
                auto xfmt = flag_zero ? "%0*X" : "%*X";
                streamf(out, xfmt, field_width, val);
                break;
            }
            case 'Y':
                va_arg(ap, const Streamable*)->stream(out);
                break;
            default:
                throw std::invalid_argument(std::string("unknown format specifier: ") + *(f - 1));
        }
    }
    out << fmt; // stream rest
}

}
