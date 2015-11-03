#include "thorin/util/stream.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "thorin/util/log.h"

namespace thorin {

unsigned int indent::level = 0;

std::string Streamable::to_string() const {
    std::ostringstream out;
    stream(out);
    return out.str();
}

void Streamable::dump() const { stream(std::cout) << thorin::endl; }
std::ostream& operator << (std::ostream& ostream, const Streamable* s) { return s->stream(ostream); }

std::ostream& streamf(std::ostream& out, const char* fmt) {
    while (*fmt) {
        if (*fmt == '%') {
            if (*(fmt+1) == '%')
                ++fmt;
            else
                throw std::invalid_argument("invalid format string: missing arguments");
        }
        out << *fmt++;
    }
    return out;
}

}
