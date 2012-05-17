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

//------------------------------------------------------------------------------

bool PrimOp::compare(PrimOp* other) const {
    if (this->hash() != other->hash())
        return false;

    if (typeid(*this) != typeid(*other))
        return false;

    if (this->index() != other->index())
        return false;

    if (const ArithOp* a = this->isa<ArithOp>()) {
        const ArithOp* b = other->as<ArithOp>();

        if (a->luse.def() != b->luse.def())
            return false;

        if (a->luse.def() != b->ruse.def())
            return false;

        return false;
    }

    ANYDSL_UNREACHABLE;
}

//------------------------------------------------------------------------------

} // namespace anydsl
