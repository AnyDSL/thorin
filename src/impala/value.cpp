#include "impala/value.h"

#include "impala/binding.h"

#if 0

using anydsl::Undef;
using anydsl::dcast;

namespace impala {

anydsl::Def* Value::load() {
    if (kind == RVALUE) {
        if (dcast<Undef>(def))
            def->error() << "value undefined\n";

        return def;
    }

    if (dcast<Undef>(bind->def))
        bind->error() << "the value of variable '" << bind->sym.str() << "' is undefined\n";

    return bind->def;
}

void Value::store(anydsl::Def* newDef) {
    if (kind == RVALUE)
        def->error() << "lvalue required\n";
    else
        bind->def = newDef;
}

const anydsl::Location& Value::loc() const {
    if (kind == RVALUE)
        return def->loc();
    else
        return bind->def->loc();
}

anydsl::Type* Value::type() {
    if (kind == RVALUE)
        return def->type();
    else
        return bind->def->type();
}

} // namespace impala

#endif
