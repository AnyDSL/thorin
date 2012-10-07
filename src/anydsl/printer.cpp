#include "anydsl/printer.h"

#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

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
    if (fancy()) {
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

    if (!def->debug.empty())
        o << def->debug;
    else
        o << def;

    if (fancy())
        o << "\33[m";

    return *this;
}

} // namespace anydsl2
