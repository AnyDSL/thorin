#include "thorin/util/printer.h"

namespace thorin {

std::ostream& Printer::newline() {
    stream_ << std::endl;
    for (int i = 0; i < indent; ++i)
        stream_ << "    ";
    return stream();
}

std::ostream& Printer::color(int c) {
    if (colored_)
        stream() << "\33[" << c << "m";
    return stream();
}

std::ostream& Printer::reset_color() {
    if (colored_)
        stream() << "\33[m";
    return stream();
}

}
