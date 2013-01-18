#include "anydsl2/printer.h"

#include <boost/cstdint.hpp>

#include "anydsl2/def.h"

namespace anydsl2 {

Printer& Printer::newline() {
    o << '\n';
    for (int i = 0; i < indent; ++i)
        o << "    ";

    return *this;
}

Printer& Printer::up() {
    ++indent;
    return newline();
}

Printer& Printer::down() {
    --indent;
    return newline();
}

Printer& Printer::dump_name(const Def* def) {
    if (fancy_) {
        unsigned i = uintptr_t(def);
        unsigned sum = 0;

        while (i) {
            sum += i & 0x3;
            i >>= 2;
        }

        sum += i;

        // elide white = 0 and black = 7
        int code = (sum % 6) + 30 + 1;
        o << "\33[" << code << "m";
    }

    if (!def->name.empty())
        o << def->name;
    else
        o << (void*)def;

    if (fancy_)
        o << "\33[m";
    return *this;
}

} // namespace anydsl2
