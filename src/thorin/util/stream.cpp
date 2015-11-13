#include "thorin/util/stream.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace thorin {

unsigned int detail::indent = 0;

std::string Streamable::to_string() const {
    std::ostringstream os;
    stream(os);
    return os.str();
}

void Streamable::dump() const { stream(std::cout) << thorin::endl; }
std::ostream& operator << (std::ostream& ostream, const Streamable* s) { return s->stream(ostream); }

std::ostream& streamf(std::ostream& os, const char* fmt) {
    while (*fmt) {
        if (*fmt == '%')
            throw std::invalid_argument("invalid format string: missing arguments");
        os << *fmt++;
    }
    return os;
}

}
