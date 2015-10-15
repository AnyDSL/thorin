#include "log.h"

#include <cstdlib>
#include <cstring>
#include <ios>
#include <iostream>
#include <new>
#include <stdexcept>
#include <sstream>

#include "streamf.h"

namespace thorin {

void Streamable::dump() const { stream(std::cout) << std::endl;; }

std::string Streamable::to_string() const {
    std::ostringstream out;
    stream(out);
    return out.str();
}

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
