// #define _GNU_SOURCE /* See feature_test_macros(7) */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <ostream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ios>
#include <new>
#include <stdexcept>

#include "log.h"

namespace thorin {

Log::Level Log::level_ = Log::Info;
std::ostream* Log::stream_ = nullptr;

/*static void fpututf32(utf32 const c, FILE *const out) {
if (c < 0x80U) {
fputc(c, out);
} else if (c < 0x800) {
fputc(0xC0 | (c >> 6), out);
fputc(0x80 | (c & 0x3F), out);
} else if (c < 0x10000) {
fputc(0xE0 | ( c >> 12), out);
fputc(0x80 | ((c >>  6) & 0x3F), out);
fputc(0x80 | ( c        & 0x3F), out);
} else {
fputc(0xF0 | ( c >> 18), out);
fputc(0x80 | ((c >> 12) & 0x3F), out);
fputc(0x80 | ((c >>  6) & 0x3F), out);
fputc(0x80 | ( c        & 0x3F), out);
}
}*/


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

void messagevf(std::ostream& out, char const *fmt, va_list ap) {
    //FILE *const out = stderr;

    for (char const *f; (f = strchr(fmt, '%')); fmt = f) {
        for(unsigned int i = 0; i < f - fmt; i++) {
            out << fmt[i];
        }

        //fwrite(fmt, sizeof(*fmt), f - fmt, out); // Print till '%'.
        ++f; // Skip '%'.

        bool extended = false;
        bool flag_zero = false;
        bool flag_long = false;
        bool flag_high = false;
        for (; ; ++f) {
            switch (*f) {
                case '#':
                    extended = true;
                    break;
                case '0':
                    flag_zero = true;
                    break;
                case 'l':
                    flag_long = true;
                    break;
                case 'h':
                    flag_high = true;
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
        char const *const rest = strstart(f, ".*");
        if (rest) {
            f = rest;
            precision = va_arg(ap, int);
        }

        /* Automatic highlight for some formats. */
        /*if (!flag_high)
            flag_high = strchr("EKNQTYk", *f);

        if (flag_high)
            fputs(colors.highlight, out);*/
        switch (*f++) {
            case '%':
                out << '%';
                //fputc('%', out);
                break;

            case 'c': {
                /*if (flag_long) {
                    const utf32 val = va_arg(ap, utf32);
                    fpututf32(val, out);
                } else {*/
                    const unsigned char val = (unsigned char) va_arg(ap, int);
                    out << val;
                //}
                break;
            }
            
            case 'i':
            case 'd': {
                if (flag_long) {
                    const long val = va_arg(ap, long);
                    streamf(out, "%ld", val);
                } else {
                    const int val = va_arg(ap, int);
                    streamf(out, "%d", val);
                }
                break;
            }

            case 's': {
                const char* const str = va_arg(ap, const char*);
                streamf(out, "%.*s", precision, str);
                break;
            }

            case 'S': {
                const std::string *str = va_arg(ap, const std::string*);
		//const string_t *str = va_arg(ap, const string_t*);
                out << str;
                //fwrite(str->begin, 1, str->size, out);
                break;
            }

            case 'u': {
                const unsigned int val = va_arg(ap, unsigned int);
                streamf(out, "%u", val);
                break;
            }

            case 'X': {
                unsigned int const val = va_arg(ap, unsigned
                        int);
                char const *const xfmt = flag_zero ? "%0*X" : "%*X";
                streamf(out, xfmt, field_width, val);
                break;
            }

            case 'Y': {
                const Streamable* s = va_arg(ap, const Streamable*);
                s->stream(out);
                break;
            }

            default:
                throw std::invalid_argument(std::string("unknown format specifier: ") + *(f - 1));
        }
        /*if (flag_high)
            fputs(colors.reset_highlight, out);*/
    }
    out << fmt; // Print rest.
}

void Log::log(Log::Level level, const char* fmt, ...) {
	if (Log::stream_ && level <= Log::level()) {
		va_list ap;
		va_start(ap, fmt);
		messagevf(Log::stream(), fmt, ap);
		va_end(ap);
	}
}

}
