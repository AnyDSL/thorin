#include "anydsl2/util/printer.h"

namespace anydsl2 {

std::ostream& Printer::newline() {
    stream_ << '\n';
    for (int i = 0; i < indent; ++i)
        stream_ << "    ";
    return stream();
}

} // namespace anydsl2
