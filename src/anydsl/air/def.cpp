#include "anydsl/air/def.h"

#include <typeinfo>

#include "anydsl/air/binop.h"
#include "anydsl/air/type.h"
#include "anydsl/air/use.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

void Def::registerUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) == uses_.end(), "must not be inside the use list");
    uses_.insert(use);
}

void Def::unregisterUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) != uses_.end(), "must be inside the use list");
    uses_.erase(use);
}

World& Def::world() const { 
    return type_->world(); 
}

} // namespace anydsl
