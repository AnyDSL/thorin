#include "anydsl2/printer.h"

#include "anydsl2/node.h"

namespace anydsl2 {

Printer& Printer::newline() {
    o << '\n';
    for (int i = 0; i < indent; ++i)
        o << "    ";

    return *this;
}

Printer& Printer::operator << (const Node* n) {
    if (n)
        n->print(*this);
    else
        o << "<NULL>";

    return *this;
}

} // namespace anydsl2
