#ifndef THORIN_UTIL_STREAMF_H
#define THORIN_UTIL_STREAMF_H

#include <cstdarg>
#include <ostream>

namespace thorin {

/// Inherit from this class and implement @p stream in order to use @p streamf.
class Streamable {
public:
    virtual std::ostream& stream(std::ostream&) const = 0;
    void dump() const; ///< Uses @p stream in order to dump to @p std::cout.
    std::string to_string() const; ///< Uses @p stream and @c std::ostringstream to generate a @c std::string.
};

/// Use @p Streamable in C++ streams via @c operator<<.
std::ostream& operator << (std::ostream&, const Streamable*);

namespace detail {
    template<typename T> inline std::ostream& stream(std::ostream& out, T val) { return out << val; }
    template<> inline std::ostream& stream<const Streamable*>(std::ostream& out, const Streamable* s) { return s->stream(out); }
}

/// Base case.
std::ostream& streamf(std::ostream& out, const char* fmt);

/** 
 * fprintf-like function which works on C++ @c std::ostream.
 * Each @c "%" in @p fmt corresponds to one vardiac argument in @p args.
 * The type of the corresponding argument must either support @c operator<< for C++ @c std::ostream or inherit from @p Streamable.
 * Use @c "%%" to escape.
 */
template<typename T, typename... Args>
std::ostream&streamf(std::ostream& out, const char* fmt, T val, Args... args) {
    while (*fmt) {
        if (*fmt == '%') {
            if (*(fmt+1) == '%')
                ++fmt;
            else
                return streamf(detail::stream(out, val), ++fmt, args...); // call even when *fmt == 0 to detect extra arguments
        }
        out << *fmt++;
    }
    return out;
}

}

#endif
