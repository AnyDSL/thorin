#ifndef THORIN_STREAM_H
#define THORIN_STREAM_H

#include <cassert>

#include "thorin/def.h"

namespace thorin {

class Def;

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
    template<class T> Stream& operator<<(const T& t) { os_ << t; return *this; }

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

    template<typename T, typename... Args>
    Stream& streamf(const char* fmt, T val, Args... args) {
        //using thorin::operator<<;

        while (*fmt) {
            auto next = fmt + 1;
            if (*fmt == '{') {
                if (*next == '{') {
                    (*this) << '{';
                    fmt += 2;
                    continue;
                }

                fmt++; // skip opening brace '{'
                if (*fmt != '}') throw std::invalid_argument("unmatched closing brace '}' in format string");
                (*this) << val;
                ++fmt; // skip closing brace '}'
                return streamf(fmt, args...); // call even when *fmt == 0 to detect extra arguments
            } else if (*fmt == '}') {
                if (*next == '}') {
                    (*this) << '}';
                    fmt += 2;
                    continue;
                }
                throw std::invalid_argument("unmatched/unescaped closing brace '}' in format string");
            } else {
                (*this) << *fmt++;
            }
        }
        throw std::invalid_argument("invalid format string for 'streamf': runaway arguments; use 'catch throw' in 'gdb'");
    }

    /// Base case.
    Stream& streamf(const char* fmt) {
        while (*fmt) {
            auto next = fmt + 1;
            if (*fmt == '{') {
                if (*next == '{') {
                    (*this) << '{';
                    fmt += 2;
                    continue;
                }

                while (*fmt && *fmt != '}') fmt++;

                if (*fmt == '}')
                    throw std::invalid_argument("invalid format string for 'streamf': missing argument(s); use 'catch throw' in 'gdb'");
                else
                    throw std::invalid_argument("invalid format string for 'streamf': missing closing brace and argument; use 'catch throw' in 'gdb'");
            } else if (*fmt == '}') {
                if (*next == '}') {
                    (*this) << '}';
                    fmt += 2;
                    continue;
                }
                throw std::invalid_argument("unmatched/unescaped closing brace '}' in format string");
            }
            (*this) << *fmt++;
        }
        return *this;
    }
    //@}

private:
    std::ostream& os_;
    std::string tab_;
    size_t level_;
};

enum class Recurse { No, OneLevel };

Stream& stream(Stream&, const Def*, Recurse recurse);
Stream& stream_assignment(Stream&, const Def*);

}

#endif
