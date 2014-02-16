#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope& scope, ArrayRef<Def> drop, ArrayRef<Def> lift, const GenericMap& generic_map = GenericMap());
inline Lambda* drop(const Scope& scope, ArrayRef<Def> with, const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, with, Array<Def>(), generic_map);
}
inline Lambda* clone(const Scope& scope, const GenericMap& generic_map = GenericMap()) { 
    return mangle(scope, Array<Def>(scope.entry()->num_params()), Array<Def>(), generic_map);
}
inline Lambda* lift(const Scope& scope, ArrayRef<Def> what, const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, Array<Def>(scope.entry()->num_params()), what, generic_map);
}

Lambda* drop_stub(Def2Def& old2new, Lambda* oentry, ArrayRef<Def> drop, const GenericMap& generic_map);

}

#endif
