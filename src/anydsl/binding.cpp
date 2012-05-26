#include "anydsl/binding.h"

#include "anydsl/util/assert.h"

using namespace anydsl;

namespace anydsl {

std::ostream& Binding::error() const { 
    //return def->error(); 
    return std::cerr;
}

} // namespace anydsl
