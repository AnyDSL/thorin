#ifndef THORIN_UTIL_PRINTER_H
#define THORIN_UTIL_PRINTER_H

#include <iostream>
#include <iomanip>
#include <string>

namespace thorin {

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
    std::ostream& dump_list(Emit emit, const List& list, const char* begin = "", const char* end = "", const char* sep = ", ");
    std::ostream& stream() { return stream_; }
    std::ostream& color(int c) { return stream() << "\33[" << c << "m"; }
    std::ostream& reset_color() { return stream() << "\33[m"; }

    int indent;

protected:
    std::ostream& stream_;

private:
    bool fancy_;
};

template<class Emit, class List>
std::ostream& Printer::dump_list(Emit emit, const List& list, const char* begin, const char* end, const char* sep) {
    stream() << begin;
    const char* separator = "";
    for (auto elem : list) {
        stream() << separator;
        emit(elem);
        separator = sep;
    }
    return stream() << end;
}

} // namespace thorin

#endif
