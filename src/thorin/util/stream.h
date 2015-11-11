#ifndef THORIN_UTIL_STREAM_H
#define THORIN_UTIL_STREAM_H

#include <ostream>

namespace thorin {

/// Inherit from this class and implement @p stream in order to use @p streamf.
class Streamable {
public:
    virtual ~Streamable() {}

    virtual std::ostream& stream(std::ostream&) const = 0;
    std::string to_string() const; ///< Uses @p stream and @c std::ostringstream to generate a @c std::string.
    void dump() const; ///< Uses @p stream in order to dump to @p std::cout.
};

std::ostream& operator << (std::ostream&, const Streamable*); ///< Use @p Streamable in C++ streams via @c operator<<.

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
std::ostream& streamf(std::ostream& out, const char* fmt, T val, Args... args) {
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

template<class Emit, class List>
std::ostream& stream_list(std::ostream& out, Emit emit, const List& list, const char* begin, const char* end, const char* sep, bool nl) {
    out << begin;
    const char* cur_sep = "";
    bool cur_nl = false;
    for (const auto& elem : list) {
        out << cur_sep;
        if (cur_nl)
            out << thorin::endl;
        emit(elem);
        cur_sep = sep;
        cur_nl = true & nl;
    }
    return out << end;
}

namespace detail {
  extern unsigned int indent;
}

template <class charT, class traits>
std::basic_ostream<charT,traits>& endl(std::basic_ostream<charT,traits>& os) {
    os << std::endl;
    os << std::string('\t', detail::indent);
    return os;
}

template <class charT, class traits>
std::basic_ostream<charT,traits>& up(std::basic_ostream<charT,traits>& os) {
    detail::indent++;
    return os;
}

template <class charT, class traits>
std::basic_ostream<charT,traits>& down(std::basic_ostream<charT,traits>& os) {
    detail::indent--;
    return os;
}

}

#endif
