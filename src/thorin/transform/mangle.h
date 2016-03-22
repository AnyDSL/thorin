#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope&, Defs args, Defs lift);

inline Lambda* drop(const Scope& scope, Defs args) {
    return mangle(scope, args, Array<const Def*>());
}

Continuation* drop(const Call&);

inline Lambda* lift(const Scope& scope, Defs defs) {
    return mangle(scope, Array<const Def*>(scope.entry()->num_params()), defs);
}

inline Lambda* clone(const Scope& scope) {
    return mangle(scope, Array<const Def*>(scope.entry()->num_params()), Defs());
}

}

#endif
