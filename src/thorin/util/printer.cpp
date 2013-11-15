#include "thorin/util/printer.h"

namespace thorin {

std::ostream& Printer::newline() {
    stream_ << '\n';
    for (int i = 0; i < indent; ++i)
        stream_ << "    ";
    return stream();
}

std::ostream& Printer::color(int c) {
    if (colored_)
        return stream() << "\33[" << c << "m";
    else
        return stream();
}

std::ostream& Printer::reset_color() {
    if (colored_)
        return stream() << "\33[m";
    else
        return stream();
}

} // namespace thorin
