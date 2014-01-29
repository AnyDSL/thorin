#ifndef THORIN_UTIL_PRINTER_H
#define THORIN_UTIL_PRINTER_H

#include <iostream>
#include <iomanip>
#include <string>

namespace thorin {

class Printer {
public:
    Printer(std::ostream& stream, bool fancy = true, bool colored = true)
        : indent(0)
        , stream_(stream)
        , fancy_(fancy)
        , colored_(colored)
    {}

    bool is_fancy() const { return fancy_; }
    bool is_colored() const { return colored_; }
    std::ostream& newline();
    std::ostream& up()   { ++indent; return newline(); }
    std::ostream& down() { --indent; return newline(); }
    template<class Emit, class List>
    std::ostream& dump_list(Emit emit, const List& list, 
            const char* begin = "", const char* end = "", const char* sep = ", ", bool nl = false);
    std::ostream& stream() { return stream_; }
    std::ostream& color(int c);
    std::ostream& reset_color();

    int indent;

protected:
    std::ostream& stream_;

private:
    bool fancy_;
    bool colored_;
};

template<class Emit, class List>
std::ostream& Printer::dump_list(Emit emit, const List& list, 
        const char* begin, const char* end, const char* sep, bool nl) {
    stream() << begin;
    const char* cur_sep = "";
    bool cur_nl = false;
    for (const auto& elem : list) {
        stream() << cur_sep;
        if (cur_nl) newline();
        emit(elem);
        cur_sep = sep;
        cur_nl = true & nl;
    }
    return stream() << end;
}

} // namespace thorin

#endif
