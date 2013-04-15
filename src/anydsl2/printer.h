#ifndef ANYDSL2_PRINTER_H
#define ANYDSL2_PRINTER_H

#include <iostream>

namespace anydsl2 {

class Node;

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , indent(0)
        , fancy_(fancy)
    {}

    bool is_fancy() const { return fancy_; }

    Printer& newline();
    Printer& up()   { ++indent; return newline(); }
    Printer& down() { --indent; return newline(); }
    Printer& operator << (const Node* def);
    Printer& operator << (const char* data) { o << data; return *this; }
    Printer& operator << (const std::string& data) { o << data; return *this; }
    Printer& operator << (size_t data) { o << data; return *this; }

    std::ostream& o;
    int indent;

private:

    bool fancy_;
};

#define ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, begin, list, end) { \
        (p) << (begin); \
        const char* sep = ""; \
        for_all (elem, (list)) { \
            (p) << sep << elem; \
            sep = ", "; \
        } \
        (p) << (end); \
    }

#define ANYDSL2_DUMP_COMMA_LIST(p, list) ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, "", list, "")

} // namespace anydsl2

#endif
