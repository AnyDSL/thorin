#ifndef IMPALA_BINDING_H
#define IMPALA_BINDING_H

#include <map>

#include "anydsl/support/symbol.h"

namespace anydsl {
    class Beta;
    class Branch;
    class CExpr;
    class Def;
    class Fix;
    class Lambda;
    class Location;
    class Param;
    class Type;
}

namespace impala {

class BB;
class Emitter;
class Fct;
class Parser;
typedef std::set<BB*> BBList;

struct Binding {
    Binding() {}
    Binding(const anydsl::Symbol sym, anydsl::Def* def)
        : sym(sym)
        , def(def)
    {}

    bool operator < (const Binding& bind) { 
        return anydsl::Symbol::FastLess()(sym, bind.sym);
    }

    std::ostream& error() const;

    const anydsl::Symbol sym;
    anydsl::Def* def;
};

} // namespace impala

#endif // IMPALA_BINDING_H
