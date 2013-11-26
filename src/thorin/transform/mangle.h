#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"

namespace thorin {

class Scope;

Lambda* mangle(const Scope& scope, 
               Def2Def& mapping,
               ArrayRef<size_t> to_drop, 
               ArrayRef<Def> drop_with, 
               ArrayRef<Def> to_lift, 
               const GenericMap& generic_map = GenericMap());

Lambda* drop(const Scope& scope, Def2Def& mapping, ArrayRef<Def> with);

inline Lambda* drop(const Scope& scope, ArrayRef<Def> with) {
    Def2Def mapping;
    return drop(scope, mapping, with);
}

inline Lambda* clone(const Scope& scope, const GenericMap& generic_map = GenericMap()) { 
    Def2Def mapping;
    return mangle(scope, mapping, Array<size_t>(), Array<Def>(), Array<Def>(), generic_map);
}

inline Lambda* drop(const Scope& scope,
                    Def2Def& mapping,
                    ArrayRef<size_t> to_drop,
                    ArrayRef<Def> drop_with,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, mapping, to_drop, drop_with, Array<Def>(), generic_map);
}

inline Lambda* drop(const Scope& scope,
                    ArrayRef<size_t> to_drop,
                    ArrayRef<Def> drop_with,
                    const GenericMap& generic_map = GenericMap()) {
    Def2Def mapping;
    return drop(scope, mapping, to_drop, drop_with, generic_map);
}

inline Lambda* lift(const Scope& scope,
                    Def2Def& mapping,
                    ArrayRef<Def> to_lift,
                    const GenericMap& generic_map = GenericMap()) {
    return mangle(scope, mapping, Array<size_t>(), Array<Def>(), to_lift, generic_map);
}

inline Lambda* lift(const Scope& scope,
                    ArrayRef<Def> to_lift,
                    const GenericMap& generic_map = GenericMap()) {
    Def2Def mapping;
    return lift(scope, mapping, to_lift, generic_map);
}

}

#endif
