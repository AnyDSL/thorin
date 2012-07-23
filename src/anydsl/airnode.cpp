#include "anydsl/airnode.h"

#include "anydsl/type.h"

namespace anydsl {

bool AIRNode::isType() const { 
    // don't use enum magic here -- there may be user defined types
    return isa<Type>(); 
}

} // namespace anydsl
