#ifndef ANYDSL2_UTIL_PRINTER_H
#define ANYDSL2_UTIL_PRINTER_H

#include <iostream>
#include <string>

namespace anydsl2 {

class Node;

class Printer {
public:
    Printer(std::ostream& o, bool fancy)
        : indent(0)
        , o(o)
        , fancy_(fancy)
    {}

    bool is_fancy() const { return fancy_; }

    Printer& newline();
    Printer& up()   { ++indent; return newline(); }
    Printer& down() { --indent; return newline(); }
    template<class Emit, class List>
    std::ostream& dump_list(Emit emit, const List& list, const char* begin = "", const char* end = "");
    operator std::ostream& () { return o; }

    int indent;

protected:
    std::ostream& o;

private:
    bool fancy_;
};


template<class Emit, class List>
std::ostream& Printer::dump_list(Emit emit, const List& list, const char* begin, const char* end) {
    o << begin;
    const char* sep = "";
    for (auto elem : list) {
        o << sep;
        emit(elem);
        sep = ", ";
    }
    return o << end;
}

#define ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, begin, list, end) { \
        (p) << (begin); \
        const char* sep = ""; \
        for (auto elem : (list)) { \
            (p) << sep << elem; \
            sep = ", "; \
        } \
        (p) << (end); \
    }

#define ANYDSL2_DUMP_COMMA_LIST(p, list) ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, "", list, "")


} // namespace anydsl2

#endif
