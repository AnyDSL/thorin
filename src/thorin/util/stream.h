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

// TODO resort to std::fmt once we are switching to C++20
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
    Stream& endl();
    Stream& flush() { ostream().flush(); return *this; }
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
     * Finally, you can use @c '\n', '\t', and '\b' to @p endl, @p indent, or @p dedent, respectively.
     */
    template<class T, class... Args> Stream& fmt(const char* s, T&& t, Args&&... args);
    Stream& fmt(const char* s); ///< Base case.
    template<class R, class F, bool rangei = false> Stream& range(const R& r, const char* sep, F f);
    template<class R, class F, bool rangei = false> Stream& range(const R& r, F f) { return range(r, ", ", f); }
    template<class R, class F> Stream& rangei(const R& r, const char* sep, F f) { return range<R, F, true>(r, sep, f); }
    template<class R, class F> Stream& rangei(const R& r, F f) { return range<R, F, true>(r, ", ", f); }
    template<class R> Stream& range(const R& r, const char* sep = ", ") { return range(r, sep, [&](const auto& x) { (*this) << x; }); }
    //@}

private:
    bool match2nd(const char* next, const char*& s, const char c);

    std::ostream& ostream_;
    std::string tab_;
    size_t level_;
};

template<class... Args> void outf(const char* fmt, Args&&... args) { Stream(std::cout).fmt(fmt, std::forward<Args&&>(args)...).endl(); }
template<class... Args> void errf(const char* fmt, Args&&... args) { Stream(std::cerr).fmt(fmt, std::forward<Args&&>(args)...).endl(); }

template<class C>
class Streamable {
private:
    constexpr const C& child() const { return *static_cast<const C*>(this); };

public:
    /// Writes to a file with name @p filename.
    void write(const std::string& filename) const { std::ofstream ofs(filename); Stream s(ofs); child().stream(s).endl(); }
    /// Writes to a file named @c child().name().
    void write() const { write(child().name()); }
    /// Writes to stdout.
    void dump() const { Stream s(std::cout); child().stream(s).endl(); }
    /// Streams to string.
    std::string to_string() const { std::ostringstream oss; Stream s(oss); child().stream(s); return oss.str(); }
};

#define THORIN_INSTANTIATE_STREAMABLE(T)                                    \
    template<> void        Streamable<T>::write() const;                    \
    template<> void        Streamable<T>::dump() const;                     \
    template<> std::string Streamable<T>::to_string() const;

// TODO Maybe there is a nicer way to do this??? Probably, using C++20 requires ...
// I just want to find out whether "x->stream(s)" or "x.stream(s)" are valid expressions.
template<class T, class = void>  struct is_streamable_ptr                                                                               : std::false_type {};
template<class T, class = void>  struct is_streamable_ref                                                                               : std::false_type {};
template<class T>                struct is_streamable_ptr<T, std::void_t<decltype(std::declval<T>()->stream(std::declval<Stream&>()))>> : std::true_type  {};
template<class T>                struct is_streamable_ref<T, std::void_t<decltype(std::declval<T>(). stream(std::declval<Stream&>()))>> : std::true_type  {};
template<class T> static constexpr bool is_streamable_ptr_v = is_streamable_ptr<T>::value;
template<class T> static constexpr bool is_streamable_ref_v = is_streamable_ref<T>::value;

template<class T> std::enable_if_t< is_streamable_ptr_v<T>, Stream&> operator<<(Stream& s, const T& x) { return x->stream(s); }
template<class T> std::enable_if_t< is_streamable_ref_v<T>, Stream&> operator<<(Stream& s, const T& x) { return x .stream(s); }
template<class T> std::enable_if_t<!is_streamable_ptr_v<T>
                                && !is_streamable_ref_v<T>, Stream&> operator<<(Stream& s, const T& x) { s.ostream() << x; return s; } ///< Fallback uses @c std::ostream @c operator<<.

template<class T, class... Args>
Stream& Stream::fmt(const char* s, T&& t, Args&&... args) {
    while (*s != '\0') {
        auto next = s + 1;

        if (false) {
        } else if (*s == '\n') { s++; endl();
        } else if (*s == '\t') { s++; indent();
        } else if (*s == '\b') { s++; dedent();
        } else if (*s == '{') {
            if (match2nd(next, s, '{')) continue;
            s++; // skip opening brace '{'

            std::string spec;
            while (*s != '\0' && *s != '}') spec.push_back(*s++);
            assert(*s == '}' && "unmatched closing brace '}' in format string");

            if constexpr (is_range_v<T>) {
                range(t, spec.c_str());
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

template<class R, class F, bool rangei = false>
Stream& Stream::range(const R& r, const char* sep, F f) {
    const char* cur_sep = "";
    size_t j = 0;
    for (const auto& elem : r) {
        for (auto i = cur_sep; *i != '\0'; ++i) {
            if (*i == '\n')
                this->endl();
            else
                (*this) << *i;
        }
        if constexpr (rangei) {
            f(j++);
        } else {
            f(elem);
        }
        cur_sep = sep;
    }
    return *this;
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
