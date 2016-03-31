#include "thorin/util/stream.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace thorin {

namespace detail {

static unsigned int indent = 0;

void inc_indent() { indent++; }
void dec_indent() { indent--; }
unsigned int get_indent() { return indent; }

};

std::string Streamable::to_string() const {
    std::ostringstream os;
    stream(os);
    return os.str();
}

void Streamable::dump() const { stream(std::cout) << thorin::endl; }
std::ostream& operator<<(std::ostream& ostream, const Streamable* s) { return s->stream(ostream); }

std::ostream& streamf(std::ostream& os, const char* fmt) {
    while (*fmt) {
        if (*fmt == '%')
            throw std::invalid_argument("invalid format string for 'streamf': missing arguments; use 'catch throw' in 'gdb'");
        os << *fmt++;
    }
    return os;
}

}
