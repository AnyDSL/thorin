#ifndef ANYDSL_BINDING_H
#define ANYDSL_BINDING_H

#include <map>

#include "anydsl/symbol.h"

namespace anydsl {

class Def;

struct Binding {
    Binding() {}
    Binding(const Symbol sym, Def* def)
        : sym(sym)
        , def(def)
    {}

    bool operator < (const Binding& bind) { 
        return Symbol::FastLess()(sym, bind.sym);
    }

    std::ostream& error() const;

    const Symbol sym;
    Def* def;
};

} // namespace anydsl

#endif
