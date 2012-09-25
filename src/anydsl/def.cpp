#include "anydsl/def.h"

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {


//------------------------------------------------------------------------------

Def::Def(int kind, const Type* type)
    : kind_(kind)
    , type_(type)
{}

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
        set_op(i, def.op(i));
}

Def::~Def() { 
    for (size_t i = 0, e = ops().size(); i != e; ++i)
        unregister_use(i);

    for_all (use, uses_) {
        size_t i = use.index();
        anydsl_assert(use.def()->ops()[i] == this, "use points to incorrect def");
        const_cast<Def*>(use.def())->ops_[i] = 0;
    }
}

void Def::set_op(size_t i, const Def* def) {
    anydsl_assert(!op(i), "already set");
    Use use(i, this);
    anydsl_assert(def->uses_.find(use) == def->uses_.end(), "already in use set");
    def->uses_.insert(use);
    ops_[i] = def;
}

void Def::unset_op(size_t i) {
    anydsl_assert(op(i), "must be set");
    unregister_use(i);
    ops_[i] = 0;
}

void Def::unregister_use(size_t i) const {
    if (const Def* def = op(i)) {
        Use use(i, this);
        anydsl_assert(def->uses_.find(use) != def->uses_.end(), "must be inside the use set");
        def->uses_.erase(use);
    }
}

bool Def::is_const() const {
    if (node_kind() == Node_Param)
        return false;

    if (empty())
        return true;

    bool result = true;
    for (size_t i = 0, e = size(); i != e && result; ++i)
        result &= op(i)->is_const();

    return result;
}

World& Def::world() const { 
    return type_ ? type_->world() : as<Type>()->world();
}

void Def::update(size_t i, const Def* def) {
    unset_op(i);
    set_op(i, def);
}

void Def::update(ArrayRef<const Def*> defs) {
    anydsl_assert(size() == defs.size(), "sizes do not match");

    for (size_t i = 0, e = size(); i != e; ++i)
        update(i, defs[i]);
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

Lambda* Def::as_lambda() const {
    return const_cast<Lambda*>(scast<Lambda>(this)); 
}

Lambda* Def::isa_lambda() const {
    return const_cast<Lambda*>(dcast<Lambda>(this)); 
}

//------------------------------------------------------------------------------

Param::Param(const Type* type, Lambda* lambda, size_t index)
    : Def(Node_Param, type, 0)
    , lambda_(lambda)
    , index_(index)
{}

PhiOps Param::phi() const {
    size_t x = index();
    Lambda* l = lambda();
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
