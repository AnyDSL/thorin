#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Lambda* mangle(const Scope& scope, 
               ArrayRef<size_t> to_drop, 
               ArrayRef<Def> drop_with, 
               ArrayRef<Def> to_lift, 
               const GenericMap& generic_map = GenericMap());

Lambda* drop(const Scope& scope, ArrayRef<Def> with);

inline Lambda* drop(const Scope& scope, ArrayRef<Def> with) { return drop(scope, with); }

inline Lambda* clone(const Scope& scope, const GenericMap& generic_map = GenericMap()) { 
    return mangle(scope, Array<size_t>(), Array<Def>(), Array<Def>(), generic_map);
}

inline Lambda* drop(const Scope& scope,
                    ArrayRef<size_t> to_drop,
                    ArrayRef<Def> drop_with,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, to_drop, drop_with, Array<Def>(), generic_map);
}

inline Lambda* lift(const Scope& scope,
                    ArrayRef<Def> to_lift,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, Array<size_t>(), Array<Def>(), to_lift, generic_map);
}

}

#endif
