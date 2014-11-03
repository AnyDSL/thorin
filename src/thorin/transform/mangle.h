#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope&, ArrayRef<Def> drop, ArrayRef<Def> lift, const Type2Type& type2type = Type2Type());
inline Lambda* drop(const Scope& scope, ArrayRef<Def> with, const Type2Type& type2type = Type2Type()) {
    return mangle(scope, with, Array<Def>(), type2type);
}
inline Lambda* clone(const Scope& scope, const Type2Type& type2type = Type2Type()) { 
    return mangle(scope, Array<Def>(scope.entry()->num_params()), Array<Def>(), type2type);
}
inline Lambda* lift(const Scope& scope, ArrayRef<Def> what, const Type2Type& type2type = Type2Type()) {
    return mangle(scope, Array<Def>(scope.entry()->num_params()), what, type2type);
}

}

#endif
