#include "impala/value.h"

#include "anydsl/literal.h"
#include "anydsl/binding.h"
#include "anydsl/util/location.h"

using anydsl::Location;
using anydsl::Undef;

namespace impala {

const anydsl::Def* Value::load() {
    if (kind == RVALUE) {
        if (def->isa<Undef>())
            /*def->error()*/ std::cerr << "value undefined\n";

        return def;
    }

    if (bind->def->isa<Undef>())
        bind->error() << "the value of variable '" << bind->sym.str() << "' is undefined\n";

    return bind->def;
}

void Value::store(const anydsl::Def* newDef) {
    if (kind == RVALUE)
        /*def->error()*/ std::cerr << "lvalue required\n";
    else
        bind->def = newDef;
}

const Location Value::loc() const {
    if (kind == RVALUE)
        return Location(def->debug);
    else
        return Location(bind->def->debug);
}

const anydsl::Type* Value::type() {
    if (kind == RVALUE)
        return def->type();
    else
        return bind->def->type();
}

} // namespace impala
