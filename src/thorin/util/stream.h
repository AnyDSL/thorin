#ifndef THORIN_UTIL_STREAM_H
#define THORIN_UTIL_STREAM_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace thorin {

class Stream {
public:
    Stream(std::ostream& os, const std::string& tab = {"    "}, size_t level = 0)
        : os_(os)
        , tab_(tab)
        , level_(level)
    {}

    /// @name getters
    //@{
    std::ostream& ostream() { return os_; }
    std::string tab() const { return tab_; }
    size_t level() const { return level_; }
    //@}
    /// @name modify Stream
    //@{
    Stream& indent() { ++level_; return *this; }
    Stream& dedent() { assert(level_ > 0); --level_; return *this; }
    Stream& endl();
    //@}
    /// @name stream
    //@{
    template<class T> Stream& operator<<(const T& t) { ostream() << t; return *this; }

    template<class Emit, class List>
    Stream& list(const List& list, Emit emit, const char* begin = "", const char* end = "", const char* sep = ", ", bool nl = false) {
        (*this) << begin;
        const char* cur_sep = "";
        bool cur_nl = false;
        for (const auto& elem : list) {
            (*this) << cur_sep;
            if (cur_nl) endl();
            emit(elem);
            cur_sep = sep;
            cur_nl = true & nl;
        }
        return (*this) << end;
    }

    Stream& fmt(const char* s) {
#ifndef NDEBUG
        auto ptr = s;
        while (auto p = strchr(ptr, '{')) {
            if (*(p + 1) != '{') {
                assert(false && "Some symbols have not been formatted");
                break;
            }
            ptr = p + 2;
        }
#endif
        return (*this) << s;
    }

    template<class T, class... Args>
    Stream& fmt(const char* s, T&& t, Args&&... args) {
        auto ptr = s;
        auto p = strchr(ptr, '{');
        while (p && *(p + 1) == '{') p = strchr(p + 2, '{');
        assert(p != nullptr && "Missing argument to format");
        ostream().write(ptr, p - ptr);
        (*this) << t;
        return fmt(strchr(p, '}') + 1, std::forward<Args&&>(args)...);
    }
    //@}

private:
    std::ostream& os_;
    std::string tab_;
    size_t level_;
};

template<class P>
class Streamable {
private:
    constexpr const P& parent() const { return *static_cast<const P*>(this); };

public:
    /// Writes to a file with name @p filename.
    void write(const std::string& filename) const { std::ofstream ofs(filename); Stream s(ofs); parent().stream(s).endl(); }
    /// Writes to a file named @p parent().name().
    void write() const { write(parent().name()); }
    /// Writes to std out.
    void dump() const { Stream s(std::cout); parent().stream(s).endl(); }
    /// Streams to string.
    std::string to_string() const { std::ostringstream oss; Stream s(oss); parent().stream(s); return oss.str(); }
};

#ifdef NDEBUG
#define assertf(condition, ...) do { (void)sizeof(condition); } while (false)
#else
#define assertf(condition, ...)                                                                                                         \
    do {                                                                                                                                \
        if (!(condition)) {                                                                                                             \
            Stream(std::cerr).fmt("assertion '{}' failed in {}:{}: ", #condition, __FILE__,  __LINE__).fmt(__VA_ARGS__).endl(); \
            std::abort();                                                                                                               \
        }                                                                                                                               \
    } while (false)
#endif

}

#endif
