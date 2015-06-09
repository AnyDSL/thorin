#include "thorin/util/printer.h"

namespace thorin {

std::ostream& Printer::newline() {
    stream_ << std::endl;
    for (int i = 0; i < indent; ++i)
        stream_ << "    ";
    return stream();
}

}
