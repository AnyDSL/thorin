#ifndef ANYDSL_PRINTER_H
#define ANYDSL_PRINTER_H

#include <iostream>

namespace anydsl {

class Def;

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , fancy_(fancy)
        , indent_(0)
        , depth_(0)
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

private:

    bool fancy_;
    int indent_;
    int depth_;
};

} // namespace anydsl

#endif
