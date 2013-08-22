#include "anydsl2/def.h"

#include <algorithm>
#include <sstream>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/be/air.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

void Tracker::set(const Def* def) {
    release();
    def_ = def;
    assert(std::find(def->trackers_.begin(), def->trackers_.end(), this) == def->trackers_.end() && "already in trackers set");
    def->trackers_.push_back(this);
}

void Tracker::release() {
    if (def_) {
        auto i = std::find(def_->trackers_.begin(), def_->trackers_.end(), this);
        assert(i != def_->trackers_.end() && "must be in trackers set");
        *i = def_->trackers_.back();
        def_->trackers_.pop_back();
    }
}

//------------------------------------------------------------------------------

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(std::find(def->uses_.begin(), def->uses_.end(), Use(i, this)) == def->uses_.end() && "already in use set");
    def->uses_.push_back(Use(i, this));
    set(i, def);
    if (isa<PrimOp>())
        is_const_ &= def->is_const();
}

void Def::unset_op(size_t i) {
    assert(op(i) && "must be set");
    unregister_use(i);
    set(i, nullptr);
}

void Def::unset_ops() {
    for (size_t i = 0, e = size(); i != e; ++i)
        unset_op(i);
}

void Def::unregister_use(size_t i) const {
    const Def* def = op(i);
    auto it = std::find(def->uses_.begin(), def->uses_.end(), Use(i, this));
    assert(it != def->uses_.end() && "must be in use set");
    *it = def->uses_.back();
    def->uses_.pop_back();
}

std::string Def::unique_name() const {
    std::ostringstream oss;
    oss << name << '_' << gid();
    return oss.str();
}

Array<Use> Def::copy_uses() const {
    Array<Use> result(uses().size());
    std::copy(uses().begin(), uses().end(), result.begin());
    return result;
}

AutoVector<const Tracker*> Def::tracked_uses() const {
    AutoVector<const Tracker*> result(uses().size());
    for (size_t i = 0, e = uses().size(); i != e; ++i)
        result[i] = new Tracker(uses()[i]);

    return result;
}

std::vector<MultiUse> Def::multi_uses() const {
    std::vector<MultiUse> result;
    auto uses = copy_uses();
    std::sort(uses.begin(), uses.end());

    const Def* cur = nullptr;
    for (auto use : uses) {
        if (cur != use.def()) {
            result.push_back(use);
            cur = use.def();
        } else
            result.back().append_user(use.index());
    }

    return result;
}

bool Def::is_primlit(int val) const {
    if (auto lit = this->isa<PrimLit>()) {
        Box box = lit->value(); // TODO
        switch (lit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return box.get_##T() == T(val);
#include "anydsl2/tables/primtypetable.h"
        }
    }

    if (auto vector = this->isa<Vector>()) {
        for (auto op : vector->ops()) {
            if (!op->is_primlit(val))
                return false;
        }

        return true;
    }

    return false;
}

bool Def::is_minus_zero() const {
    if (auto lit = this->isa<PrimLit>()) {
        Box box = lit->value();
        switch (lit->primtype_kind()) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return box.get_##T() == T(0);
#include "anydsl2/tables/primtypetable.h"
            case PrimType_f32: return box.get_f32() == -0.f;
            case PrimType_f64: return box.get_f64() == -0.0;
        }
    }
    return false;
}

void Def::replace(const Def* with) const {
    for (auto i = trackers_.begin(), e = trackers_.end(); i != e;)
        **(i++) = with;

    auto uses = multi_uses();

    for (auto use : uses) {
        if (auto lambda = use->isa_lambda()) {
            for (auto index : use.indices())
                lambda->update_op(index, with);
        } else {
            PrimOp* released = world().release(use->as<PrimOp>());
            for (auto index : use.indices())
                released->update(index, with);
        }
    }

    for (auto use : uses) {
        if (auto oprimop = (PrimOp*) use->isa<PrimOp>()) {
            Array<const Def*> ops(oprimop->ops());
            for (auto index : use.indices())
                ops[index] = with;
            size_t old_gid = world().gid();
            const Def* ndef = world().rebuild(oprimop, ops);

            if (oprimop->kind() == ndef->kind()) {
                assert(oprimop->size() == ndef->size());

                size_t j = 0;
                size_t index = use.index(j);
                for (size_t i = 0, e = oprimop->size(); i != e; ++i) {
                    if (i != index && oprimop->op(i) != ndef->op(i))
                        goto recurse;
                    if (i == index && j < use.num_indices())
                        index = use.index(j++);
                }

                if (ndef->gid() == old_gid) { // only consider fresh (non-CSEd) primop
                    // nothing exciting happened by rebuilding 
                    // -> reuse the old chunk of memory and save recursive updates
                    AutoPtr<PrimOp> nreleased = world().release(ndef->as<PrimOp>());
                    nreleased->unset_ops();
                    world().reinsert(oprimop);
                    continue;
                }
            }
recurse:
            oprimop->replace(ndef);
            oprimop->unset_ops();
            delete oprimop;
        }
    }
}

int Def::non_const_depth() const {
    if (this->is_const() || this->isa<Param>()) 
        return 0;

    const PrimOp* primop = this->as<PrimOp>();
    int max = 0;
    for (auto op : primop->ops()) {
        int d = op->non_const_depth();
        max = d > max ? d : max;
    }

    return max + 1;
}

void Def::dump() const { 
    auto primop = this->isa<PrimOp>();
    if (primop && !primop->is_const())
        emit_assignment(primop);
    else {
        emit_def(this); 
        std::cout << std::endl;
    }
}

World& Def::world() const { return type()->world(); }
const Def* Def::op_via_lit(const Def* def) const { return op(def->primlit_value<size_t>()); }
Lambda* Def::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* Def::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
const PrimOp* Def::is_non_const_primop() const { return is_const() ? nullptr : isa<PrimOp>(); }
int Def::order() const { return type()->order(); }
bool Def::is_generic() const { return type()->is_generic(); }
size_t Def::length() const { return type()->as<VectorType>()->length(); }

//------------------------------------------------------------------------------

Param::Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name)
    : Def(gid, Node_Param, 0, type, false, name)
    , lambda_(lambda)
    , index_(index)
{}

Peeks Param::peek() const {
    size_t x = index();
    Lambda* l = lambda();
    Lambdas preds = l->direct_preds();
    Peeks result(preds.size());
    for (size_t i = 0, e = preds.size(); i != e; ++i)
        result[i] = Peek(preds[i]->arg(x), preds[i]);

    return result;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
