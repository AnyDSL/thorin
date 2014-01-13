#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"

namespace thorin {

class Scope;

Lambda* mangle(const Scope& scope, 
               Def2Def& old2new,
               ArrayRef<size_t> to_drop, 
               ArrayRef<Def> drop_with, 
               ArrayRef<Def> to_lift, 
               const GenericMap& generic_map = GenericMap());

Lambda* drop(const Scope& scope, Def2Def& old2new, ArrayRef<Def> with);

inline Lambda* drop(const Scope& scope, ArrayRef<Def> with) {
    Def2Def old2new;
    return drop(scope, old2new, with);
}

inline Lambda* clone(const Scope& scope, const GenericMap& generic_map = GenericMap()) { 
    Def2Def old2new;
    return mangle(scope, old2new, Array<size_t>(), Array<Def>(), Array<Def>(), generic_map);
}

inline Lambda* drop(const Scope& scope,
                    Def2Def& old2new,
                    ArrayRef<size_t> to_drop,
                    ArrayRef<Def> drop_with,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, old2new, to_drop, drop_with, Array<Def>(), generic_map);
}

inline Lambda* drop(const Scope& scope,
                    ArrayRef<size_t> to_drop,
                    ArrayRef<Def> drop_with,
                    const GenericMap& generic_map = GenericMap()) {
    Def2Def old2new;
    return drop(scope, old2new, to_drop, drop_with, generic_map);
}

inline Lambda* lift(const Scope& scope,
                    Def2Def& old2new,
                    ArrayRef<Def> to_lift,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, old2new, Array<size_t>(), Array<Def>(), to_lift, generic_map);
}

inline Lambda* lift(const Scope& scope,
                    ArrayRef<Def> to_lift,
                    const GenericMap& generic_map = GenericMap()) {
    Def2Def old2new;
    return lift(scope, old2new, to_lift, generic_map);
}

}

#endif
