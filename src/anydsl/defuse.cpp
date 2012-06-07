#include "anydsl/defuse.h"

#include <typeinfo>

#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/for_all.h"

namespace anydsl {


//------------------------------------------------------------------------------

Use::Use(AIRNode* parent, const Def* def)
    : AIRNode(Index_Use)
    , def_(def) 
    , parent_(parent)
{
    def_->registerUse(this);
}

Use::~Use() {
    def_->unregisterUse(this);
}

World& Use::world() {
    return def_->world();
}

//------------------------------------------------------------------------------

Def::~Def() { 
    anydsl_assert(isa<Lambda>() 
            || (isa<Sigma>() && as<Sigma>()->named()) 
            || uses_.empty(), 
            "there are still uses pointing to this def"); 

    for_all (const& use, ops()) use.~Use();
    ::operator delete(ops_); 
}


void Def::registerUse(Use* use) const {
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) == uses_.end(), "must not be inside the use list");
    uses_.insert(use);
}

void Def::unregisterUse(Use* use) const {
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) != uses_.end(), "must be inside the use list");
    uses_.erase(use);
}

World& Def::world() const { 
    if (type_)
        return type_->world(); 
    else 
        return as<Type>()->world();
}

bool Value::equal(const Value* other) const {
    if (this->index() != other->index())
        return false;

    if (this->numOps() != other->numOps())
        return false;

    bool result = true;
    for (size_t i = 0, e = numOps(); i != e && result; ++i)
        result &= this->ops_[i].def() == other->ops_[i].def();

    return result;
}

size_t Value::hash() const {
    size_t seed = 0;

    boost::hash_combine(seed, index());
    boost::hash_combine(seed, numOps());

    for (size_t i = 0, e = numOps(); i != e; ++i)
        boost::hash_combine(seed, ops_[i].def());

    return seed;
}

//------------------------------------------------------------------------------

Params::Params(Lambda* parent, const Sigma* sigma)
    : Def(Index_Params, sigma, 0)
    , parent_(parent)
{}

const Sigma* Params::sigma() const { 
    return type()->as<Sigma>(); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
