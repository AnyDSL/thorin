#ifndef IMPALA_VALUE_H
#define IMPALA_VALUE_H

#include <boost/cstdint.hpp>

#include "anydsl/util/location.h"

namespace anydsl {
    class Binding;
    class Def;
    class Symbol;
    class Type;
}

namespace impala {

class Value {
public:

    Value() 
        : def(0) 
    {}
    explicit Value(anydsl::Def* def)
        : kind(RVALUE)
        , def(def)
    {}
    explicit Value(anydsl::Binding* bind)
        : kind(LVALUE)
        , bind(bind)
    {}

    anydsl::Def* load();
    void store(anydsl::Def* def);
    anydsl::Type* type();

    const anydsl::Location& loc() const;
    const anydsl::Position& pos1() const { return loc().pos1(); }
    const anydsl::Position& pos2() const { return loc().pos2(); }

    intptr_t getVN() { return (intptr_t) def; };

private:

    enum Kind {
        LVALUE,
        RVALUE
    };

    Kind kind;

    union {
        anydsl::Def* def;
        anydsl::Binding* bind;
    };
};

} // namespace impala

#endif // IMPALA_VALUE_H
