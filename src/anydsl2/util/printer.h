#ifndef ANYDSL2_UTIL_PRINTER_H
#define ANYDSL2_UTIL_PRINTER_H

#include <iostream>
#include <string>

namespace anydsl2 {

class Node;

class Printer {
public:
    Printer(std::ostream& stream, bool fancy)
        : indent(0)
        , stream_(stream)
        , fancy_(fancy)
    {}

    bool is_fancy() const { return fancy_; }
    std::ostream& newline();
    std::ostream& up()   { ++indent; return newline(); }
    std::ostream& down() { --indent; return newline(); }
    template<class Emit, class List>
    std::ostream& dump_list(Emit emit, const List& list, const char* begin = "", const char* end = "");
    std::ostream& stream() { return stream_; }

    int indent;

protected:
    std::ostream& stream_;

private:
    bool fancy_;
};

template<class Emit, class List>
std::ostream& Printer::dump_list(Emit emit, const List& list, const char* begin, const char* end) {
    stream_ << begin;
    const char* sep = "";
    for (auto elem : list) {
        stream_ << sep;
        emit(elem);
        sep = ", ";
    }
    return stream_ << end;
}

} // namespace anydsl2

#endif
