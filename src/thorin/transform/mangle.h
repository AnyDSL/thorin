#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope&, Types type_args, Defs args, Defs lift);

inline Lambda* drop(const Scope& scope, Types type_args, Defs args) {
    return mangle(scope, type_args, args, Array<const Def*>());
}

Lambda* drop(const Call&);

inline Lambda* lift(const Scope& scope, Types type_args, Defs defs) {
    return mangle(scope, type_args, Array<const Def*>(scope.entry()->num_params()), defs);
}

inline Lambda* clone(const Scope& scope) {
    return mangle(scope, Array<const Type*>(scope.entry()->num_type_params()),
                         Array<const Def*>(scope.entry()->num_params()), Defs());
}

}

#endif
