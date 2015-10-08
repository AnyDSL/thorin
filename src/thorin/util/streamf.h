#ifndef THORIN_UTIL_STREAMF_H
#define THORIN_UTIL_STREAMF_H

#include <cstdarg>
#include <ostream>

namespace thorin {

/// Inherit from this class and implement @p stream in order to use the '%Y' conversion specifier of @p vstreamf.
class Streamable {
public:
    virtual std::ostream& stream(std::ostream&) const = 0;
    /// Uses @p stream in order to dump to @p std::cout.
    void dump() const;
};

/// Use @p Streamable in C++ streams via operator '<<'.
std::ostream& operator << (std::ostream&, const Streamable*);

namespace detail {
    template<typename T> inline std::ostream& stream(std::ostream& out, T val) { return out << val; }
    template<> inline std::ostream& stream<const Streamable*>(std::ostream& out, const Streamable* s) { return s->stream(out); }
}

/// Base case.
void streamf(std::ostream& out, const char* fmt);

/** 
 * fprintf-like function which works on C++ @p std::ostream.
 * Each "%" in @p fmt corresponds to one vardiac argument in @p args.
 * Use "%%" to escape.
 */
template<typename T, typename... Args>
void streamf(std::ostream& out, const char* fmt, T val, Args... args) {
    while (*fmt) {
        if (*fmt == '%') {
            if (*(fmt+1) == '%')
                ++fmt;
            else
                return streamf(detail::stream(out, val), ++fmt, args...); // call even when *fmt == 0 to detect extra arguments
        }
        out << *fmt++;
    }
}

}

#endif
