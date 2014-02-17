#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope&, Def2Def& old2new, ArrayRef<Def> drop, ArrayRef<Def> lift, const GenericMap& generic_map = GenericMap());
inline Lambda* drop(const Scope& scope, Def2Def& old2new, ArrayRef<Def> with, const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, old2new, with, Array<Def>(), generic_map);
}
inline Lambda* clone(const Scope& scope, Def2Def& old2new, const GenericMap& generic_map = GenericMap()) { 
    return mangle(scope, old2new, Array<Def>(scope.entry()->num_params()), Array<Def>(), generic_map);
}
inline Lambda* lift(const Scope& scope, Def2Def& old2new, ArrayRef<Def> what, const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, old2new, Array<Def>(scope.entry()->num_params()), what, generic_map);
}

inline Lambda* mangle(const Scope& scope, ArrayRef<Def> drop, ArrayRef<Def> lift, const GenericMap& generic_map = GenericMap()) {
    Def2Def old2new;
    return mangle(scope, old2new, drop, lift, generic_map);
}
inline Lambda* drop(const Scope& scope, ArrayRef<Def> with, const GenericMap& generic_map = GenericMap()) {
    Def2Def old2new;
    return mangle(scope, old2new, with, Array<Def>(), generic_map);
}
inline Lambda* clone(const Scope& scope, const GenericMap& generic_map = GenericMap()) { 
    Def2Def old2new;
    return mangle(scope, old2new, Array<Def>(scope.entry()->num_params()), Array<Def>(), generic_map);
}
inline Lambda* lift(const Scope& scope, ArrayRef<Def> what, const GenericMap& generic_map = GenericMap()) {
    Def2Def old2new;
    return mangle(scope, old2new, Array<Def>(scope.entry()->num_params()), what, generic_map);
}

}

#endif
