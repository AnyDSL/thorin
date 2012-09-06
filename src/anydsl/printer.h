#ifndef ANYDSL_PRINTER_H
#define ANYDSL_PRINTER_H

#include <iostream>

namespace anydsl {

class Def;

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , indent(0)
        , fancy_(fancy)
    {}

    bool fancy() const { return fancy_; }

    Printer& dump(const Def* def);
    Printer& dumpName(const Def* def);

    Printer& newline();
    Printer& up();
    Printer& down();

    template<class T>
    Printer& operator << (const T& data) {
        o << data;
        return *this;
    }

    std::ostream& o;
    int indent;

private:

    bool fancy_;
};

} // namespace anydsl

#endif
