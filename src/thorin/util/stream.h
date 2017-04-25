#ifndef THORIN_UTIL_STREAM_H
#define THORIN_UTIL_STREAM_H

#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace thorin {

/// Inherit from this class and implement @p stream in order to use @p streamf.
class Streamable {
public:
    virtual ~Streamable() {}

    virtual std::ostream& stream(std::ostream& os) const = 0;
    std::string to_string() const; ///< Uses @p stream and @c std::ostringstream to generate a @c std::string.
    void dump() const; ///< Uses @p stream in order to dump to @p std::cout.
};

std::ostream& operator<<(std::ostream&, const Streamable*); ///< Use @p Streamable in C++ streams via @c operator<<.

namespace detail {
    template<typename T> inline std::ostream& stream(std::ostream& os, T val) { return os << val; }
    template<> inline std::ostream& stream<const Streamable*>(std::ostream& os, const Streamable* s) { return s->stream(os); }

    template<typename T>
    const char* handle_fmt_specifier(std::ostream& os, const char* fmt, T val) {
        fmt++; // skip opening brace {
        char specifier = *fmt;
        std::string spec_fmt;
        while (*fmt && *fmt != '}') {
            spec_fmt.push_back(*fmt++);
        }
        if (*fmt != '}')
            throw std::invalid_argument("unmatched closing brace '}' in format string");
        if (specifier == '}')
            detail::stream(os, val);
        // TODO possibly handle some format specifiers here that don't require major template trickery (e.g. floats)
        return ++fmt;
    }
}

/// Base case.
std::ostream& streamf(std::ostream& os, const char* fmt);

/**
 * fprintf-like function which works on C++ @c std::ostream.
 * Each @c "{}" in @p fmt corresponds to one of the variadic arguments in @p args.
 * The type of the corresponding argument must either support @c operator<< for C++ @c std::ostream or inherit from @p Streamable.
 */
template<typename T, typename... Args>
std::ostream& streamf(std::ostream& os, const char* fmt, T val, Args... args) {
    while (*fmt) {
        auto next = fmt + 1;
        if (*fmt == '{') {
            if (*next == '{') {
                os << '{';
                fmt += 2;
                continue;
            }
            fmt = detail::handle_fmt_specifier(os, fmt, val);
            // call even when *fmt == 0 to detect extra arguments
            return streamf(os, fmt, args...);
        } else if (*fmt == '}') {
            if (*next == '}') {
                os << '}';
                fmt += 2;
                continue;
            }
            // TODO give exact position
            throw std::invalid_argument("nmatched/unescaped closing brace '}' in format string");
        } else
            os << *fmt++;
    }
    throw std::invalid_argument("invalid format string for 'streamf': runaway arguments; use 'catch throw' in 'gdb'");
}

namespace detail {
    void inc_indent();
    void dec_indent();
    unsigned int get_indent();
}

template <class charT, class traits>
std::basic_ostream<charT,traits>& endl(std::basic_ostream<charT,traits>& os) {
    return os << std::endl << std::string(detail::get_indent() * 4, ' ');
}

template <class charT, class traits>
std::basic_ostream<charT,traits>& up(std::basic_ostream<charT,traits>& os) { detail::inc_indent(); return os; }

template <class charT, class traits>
std::basic_ostream<charT,traits>& down(std::basic_ostream<charT,traits>& os) { detail::dec_indent(); return os; }

template <class charT, class traits>
std::basic_ostream<charT,traits>& up_endl(std::basic_ostream<charT,traits>& os) { return os << up << endl; }

template <class charT, class traits>
std::basic_ostream<charT,traits>& down_endl(std::basic_ostream<charT,traits>& os) { return os << down << endl; }

template<class Emit, class List>
std::ostream& stream_list(std::ostream& os, const List& list, Emit emit,
        const char* begin = "", const char* end = "", const char* sep = ", ", bool nl = false) {
    os << begin;
    const char* cur_sep = "";
    bool cur_nl = false;
    for (const auto& elem : list) {
        os << cur_sep;
        if (cur_nl)
            os << endl;
        emit(elem);
        cur_sep = sep;
        cur_nl = true & nl;
    }
    return os << end;
}

template<class Emit, class List>
class StreamList {
public:
    StreamList(const List& list, Emit emit, const char* sep)
        : emit(emit)
        , list(list)
        , sep(sep)
    {}

    Emit emit;
    const List& list;
    const char* sep;
};

template<class Emit, class List>
std::ostream& operator<<(std::ostream& os, StreamList<Emit, List> sl) {
    return stream_list(os, sl.list, sl.emit, "", "", sl.sep);
}

template<class Emit, class List>
StreamList<Emit, List> stream_list(const List& list, Emit emit, const char* sep = ", ") {
    return StreamList<Emit, List>(list, emit, sep);
}

#ifdef NDEBUG
#   define assertf(condition, ...) do { (void)sizeof(condition); } while (false)
#else
#   define assertf(condition, ...) do { \
        if (!(condition)) { \
            std::cerr << "Assertion '" #condition "' failed in " << __FILE__ << ":" << __LINE__ << " "; \
            streamf(std::cerr, __VA_ARGS__) << std::endl; \
            std::abort(); \
        } \
    } while (false)
#endif

}

#endif
