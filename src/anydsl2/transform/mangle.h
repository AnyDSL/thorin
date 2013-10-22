#ifndef ANYDSL2_TRANSFORM_MANGLE_H
#define ANYDSL2_TRANSFORM_MANGLE_H

#include "anydsl2/type.h"

namespace anydsl2 {

class Scope;

Lambda* mangle(const Scope& scope, 
               ArrayRef<size_t> to_drop, 
               ArrayRef<const DefNode*> drop_with, 
               ArrayRef<const DefNode*> to_lift, 
               const GenericMap& generic_map = GenericMap(),
               ArrayRef<Lambda*> run = ArrayRef<Lambda*>(nullptr, 0)); 

Lambda* drop(const Scope& scope, ArrayRef<const DefNode*> with, ArrayRef<Lambda*> run = ArrayRef<Lambda*>(nullptr, 0));
inline Lambda* clone(const Scope& scope, const GenericMap& generic_map) { 
    return mangle(scope, Array<size_t>(), Array<const DefNode*>(), Array<const DefNode*>(), generic_map);
}
inline Lambda* drop(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<const DefNode*> drop_with, const GenericMap& generic_map) {
    return mangle(scope, to_drop, drop_with, Array<const DefNode*>(), generic_map);
}
inline Lambda* lift(const Scope& scope, ArrayRef<const DefNode*> to_lift, const GenericMap& generic_map) {
    return mangle(scope, Array<size_t>(), Array<const DefNode*>(), to_lift, generic_map);
}

}

#endif
