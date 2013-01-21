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
    if (fancy_) // elide white = 0 and black = 7
        o << "\33[" << (def->gid() % 6 + 30 + 1) << "m";
    o << def->name << '_' << def->delimiter() << def->gid();

    if (fancy_)
        o << "\33[m";

    return *this;
}

} // namespace anydsl2
