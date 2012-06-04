#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {


const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
