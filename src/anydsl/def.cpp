#include "anydsl/def.h"

#include <typeinfo>

#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/for_all.h"

namespace anydsl {


//------------------------------------------------------------------------------

Def::~Def() { 
    for (size_t i = 0, e = numOps(); i != e; ++i)
        if (ops_[i])
            ops_[i]->unregisterUse(i, this);

    for_all (use, uses_) {
        size_t i = use.index();
        anydsl_assert(use.def()->ops()[i] == this, "use points to incorrect def");
        use.def()->delOp(i);
    }

    delete[] ops_;
}

void Def::registerUse(size_t i, const Def* def) const {
    Use use(i, def);
    anydsl_assert(uses_.find(use) == uses_.end(), "already in use set");
    uses_.insert(use);
}

void Def::unregisterUse(size_t i, const Def* def) const {
    Use use(i, def);
    anydsl_assert(uses_.find(use) != uses_.end(), "must be inside the use set");
    uses_.erase(use);
}

bool Def::isType() const { 
    // don't use enum magic here -- there may be user defined types
    return !type_;
}

World& Def::world() const { 
    if (type_)
        return type_->world(); 
    else 
        return as<Type>()->world();
}

bool Def::equal(const Def* other) const {
    if (this->indexKind() != other->indexKind())
        return false;

    if (this->numOps() != other->numOps())
        return false;

    bool result = true;
    for (size_t i = 0, e = numOps(); i != e && result; ++i)
        result &= this->ops_[i] == other->ops_[i];

    return result;
}

size_t Def::hash() const {
    size_t seed = 0;

    boost::hash_combine(seed, indexKind());
    boost::hash_combine(seed, numOps());

    for (size_t i = 0, e = numOps(); i != e; ++i)
        boost::hash_combine(seed, ops_[i]);

    return seed;
}

void Def::alloc(size_t size) { 
    assert(ops_ == 0); 
    numOps_ = size;
    ops_ = new const Def*[size];
}

//------------------------------------------------------------------------------

Param::Param(const Type* type, const Lambda* lambda, size_t index)
    : Def(Index_Param, type, 0)
    , lambda_(lambda)
    , index_(index)
{}

PhiOps Param::phiOps() const {
    size_t x = index();
    const Lambda* l = lambda();
    LambdaSet callers = l->callers();

    PhiOps result(callers.size());

    size_t i = 0;
    for_all (caller, callers)
        result[i++] = PhiOp(caller->args()[x], caller);

    return result;
}

bool Param::equal(const Def* other) const {
    if (!Def::equal(other))
        return false;

    return index() == other->as<Param>()->index() && lambda() == other->as<Param>()->lambda();
}

size_t Param::hash() const {
    size_t seed = Def::hash();
    boost::hash_combine(seed, index());
    boost::hash_combine(seed, lambda());

    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl
