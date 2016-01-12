#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope&, ArrayRef<Type> type_args, ArrayRef<Def> args, ArrayRef<Def> lift);

inline Lambda* drop(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<Def> args) {
    return mangle(scope, type_args, args, Array<Def>());
}

Lambda* drop(const Call&);

inline Lambda* lift(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<Def> defs) {
    return mangle(scope, type_args, Array<Def>(scope.entry()->num_params()), defs);
}

inline Lambda* clone(const Scope& scope) {
    return mangle(scope, Array<Type>(scope.entry()->num_type_params()), Array<Def>(scope.entry()->num_params()), Array<Def>());
}

}

#endif
