#ifndef THORIN_UTIL_STREAM_H
#define THORIN_UTIL_STREAM_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "thorin/util/iterator.h"

namespace thorin {

class Stream {
public:
    Stream(std::ostream& ostream = std::cout, const std::string& tab = {"    "}, size_t level = 0)
        : ostream_(ostream)
        , tab_(tab)
        , level_(level)
    {}

    /// @name getters
    //@{
    std::ostream& ostream() { return ostream_; }
    std::string tab() const { return tab_; }
    size_t level() const { return level_; }
    //@}
    /// @name modify Stream
    //@{
    Stream& indent() { ++level_; return *this; }
    Stream& dedent() { assert(level_ > 0); --level_; return *this; }
    Stream& endl() {
        ostream() << '\n';
        for (size_t i = 0; i != level_; ++i) ostream() << tab_;
        return *this;
    }
    //@}
    /// @name stream
    //@{
    /**
     * fprintf-like function.
     * Each @c "{}" in @p s corresponds to one of the variadic arguments in @p args.
     * The type of the corresponding argument must support @c operator<< on @c std::ostream& or preferably @p Stream.
     * Furthermore, an argument may also be a range-based container where its elements fulfill the condition above.
     * You can specify the separator within the braces:
     @code{.cpp}
         s.fmt("({, })", list) // yields "(a, b, c)"
     @endcode
     * If you use @c {\n} as separator, it will invoke Stream::endl - keeping indentation:
     @code{.cpp}
         s.fmt("({\n})", list)
     @endcode
     */
    template<class T, class... Args>
    Stream& fmt(const char* s, T&& t, Args&&... args);
    Stream& fmt(const char* s); ///< Base case.
    //@}

private:
    bool match2nd(const char* next, const char*& s, const char c);

    std::ostream& ostream_;
    std::string tab_;
    size_t level_;
};

template<class... Args> void outf(const char* fmt, Args&&... args) { Stream().fmt(fmt, std::forward<Args&&>(args)...).endl(); }
template<class... Args> void errf(const char* fmt, Args&&... args) { Stream().fmt(fmt, std::forward<Args&&>(args)...).endl(); }

template<class P>
class Streamable {
private:
    constexpr const P& parent() const { return *static_cast<const P*>(this); };

public:
    /// Writes to a file with name @p filename.
    void write(const std::string& filename) const { std::ofstream ofs(filename); Stream s(ofs); parent().stream(s).endl(); }
    /// Writes to a file named @c parent().name().
    void write() const { write(parent().name()); }
    /// Writes to stdout.
    void dump() const { Stream s(std::cout); parent().stream(s).endl(); }
    /// Streams to string.
    std::string to_string() const { std::ostringstream oss; Stream s(oss); parent().stream(s); return oss.str(); }
};

template<class T, class = void>
struct is_streamable : std::false_type {};
template<class T>
struct is_streamable<T, std::void_t<decltype(std::declval<T>()->stream(std::declval<thorin::Stream&>()))>> : std::true_type {};
template<class T> static constexpr bool is_streamable_v = is_streamable<T>::value;

template<class T> std::enable_if_t< is_streamable_v<T>, Stream&> operator<<(Stream& s, const T& t) { return t->stream(s); }
template<class T> std::enable_if_t<!is_streamable_v<T>, Stream&> operator<<(Stream& s, const T& t) { s.ostream() << t; return s; } ///< Fallback.

template<class T, class... Args>
Stream& Stream::fmt(const char* s, T&& t, Args&&... args) {
    while (*s != '\0') {
        auto next = s + 1;

        if (*s == '{') {
            if (match2nd(next, s, '{')) continue;
            s++; // skip opening brace '{'

            std::string spec;
            while (*s != '\0' && *s != '}') spec.push_back(*s++);
            assert(*s == '}' && "unmatched closing brace '}' in format string");

            if constexpr (is_range_v<T>) {
                const char* cur_sep = "";
                for (const auto& elem : t) {
                    for (auto i = cur_sep; *i != '\0'; ++i) {
                        if (*i == '\n')
                            this->endl();
                        else
                            (*this) << *i;
                    }
                    (*this) << elem;
                    cur_sep = spec.c_str();
                }
            } else {
                (*this) << t;
            }

            ++s; // skip closing brace '}'
            return fmt(s, std::forward<Args&&>(args)...); // call even when *s == '\0' to detect extra arguments
        } else if (*s == '}') {
            if (match2nd(next, s, '}')) continue;
            assert(false && "unmatched/unescaped closing brace '}' in format string");
        } else {
            (*this) << *s++;
        }
    }

    assert(false && "invalid format string for 's'");
}

inline Stream& Stream::fmt(const char* s) {
    while (*s) {
        auto next = s + 1;
        if (*s == '{') {
            if (match2nd(next, s, '{')) continue;

            while (*s && *s != '}') s++;

            assert(*s != '}' && "invalid format string for 'streamf': missing argument(s)");
            assert(false && "invalid format string for 'streamf': missing closing brace and argument");

        } else if (*s == '}') {
            if (match2nd(next, s, '}')) continue;
            assert(false && "unmatched/unescaped closing brace '}' in format string");
        }
        (*this) << *s++;
    }
    return *this;
}

inline bool Stream::match2nd(const char* next, const char*& s, const char c) {
    if (*next == c) {
        (*this) << c;
        s += 2;
        return true;
    }
    return false;
}

#ifdef NDEBUG
#define assertf(condition, ...) do { (void)sizeof(condition); } while (false)
#else
#define assertf(condition, ...)                                                                                                 \
    do {                                                                                                                        \
        if (!(condition)) {                                                                                                     \
            Stream(std::cerr).fmt("assertion '{}' failed in {}:{}: ", #condition, __FILE__,  __LINE__).fmt(__VA_ARGS__).endl(); \
            std::abort();                                                                                                       \
        }                                                                                                                       \
    } while (false)
#endif

}

#endif
