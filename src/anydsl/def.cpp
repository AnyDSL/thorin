#include "anydsl/def.h"

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {


//------------------------------------------------------------------------------

Def::Def(int kind, const Type* type, size_t size)
    : kind_(kind)
    , type_(type)
    , ops_(size)
{}

Def::Def(const Def& def)
    : debug(def.debug)
    , kind_(def.kind())
    , type_(def.type())
    , ops_(def.size())
{
    for (size_t i = 0, e = size(); i != e; ++i)
        setOp(i, def.op(i));
}

Def::~Def() { 
    for (size_t i = 0, e = ops().size(); i != e; ++i)
        if (ops_[i])
            ops_[i]->unregisterUse(i, this);

    for_all (use, uses_) {
        size_t i = use.index();
        anydsl_assert(use.def()->ops()[i] == this, "use points to incorrect def");
        const_cast<Def*>(use.def())->ops_[i] = 0;
    }
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

void Def::update(ArrayRef<size_t> x, ArrayRef<const Def*> ops) {
    assert(x.size() == ops.size());
    size_t size = x.size();

    for (size_t i = 0; i < size; ++i) {
        size_t idx = x[i];
        const Def* def = ops[i];

        op(idx)->unregisterUse(idx, this);
        setOp(idx, def);
    }
}

World& Def::world() const { 
    if (type_)
        return type_->world(); 
    else 
        return as<Type>()->world();
}

Array<Use> Def::copy_uses() const {
    Array<Use> result(uses().size());
    std::copy(uses().begin(), uses().end(), result.begin());

    return result;
}

bool Def::equal(const Def* other) const {
    return this->kind() == other->kind() && this->ops_ == other->ops_;
}

bool Def::is_primlit(int val) const {
    if (const PrimLit* lit = this->isa<PrimLit>()) {
        Box box = lit->box();

        switch (lit->primtype_kind()) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: return box.get_##T() == T(val);
#include "anydsl/tables/primtypetable.h"
        }
    }

    return false;
}

size_t Def::hash() const {
    size_t seed = 0;

    boost::hash_combine(seed, kind());
    boost::hash_combine(seed, ops_);

    return seed;
}

void Def::alloc(size_t size) { 
    anydsl_assert(ops_.empty(), "realloc");
    ops_.~Array();
    new (&ops_) Array<const Def*>(size);
}

void Def::realloc(size_t newsize) { 
    ops_.~Array();
    new (&ops_) Array<const Def*>(newsize);
}

//------------------------------------------------------------------------------

Param::Param(const Type* type, Lambda* lambda, size_t index)
    : Def(Node_Param, type, 0)
    , lambda_(lambda)
    , index_(index)
{}

Param::~Param() {
    if (lambda_)
        lambda_->params_.erase(this);
}

PhiOps Param::phi() const {
    size_t x = index();
    const Lambda* l = lambda();
    LambdaSet preds = l->preds();

    PhiOps result(preds.size());

    size_t i = 0;
    for_all (pred, preds)
        result[i++] = PhiOp(pred->arg(x), pred);

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
