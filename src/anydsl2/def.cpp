#include "anydsl2/def.h"

#include <algorithm>
#include <sstream>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/printer.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

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
        Trackers::iterator i = std::find(def_->trackers_.begin(), def_->trackers_.end(), this);
        assert(i != def_->trackers_.end() && "must be in trackers set");
        def_->trackers_.erase(i);
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
    set(i, 0);
}

void Def::unset_ops() {
    for (size_t i = 0, e = size(); i != e; ++i)
        unset_op(i);
}

void Def::unregister_use(size_t i) const {
    const Def* def = op(i);
    Uses::iterator it = std::find(def->uses_.begin(), def->uses_.end(), Use(i, this));
    assert(it != def->uses_.end() && "must be in use set");
    def->uses_.erase(it);
}

//bool Def::is_const() const {
    //if (isa<Param>() || type()->isa<Mem>() || node_kind() == Node_Enter || node_kind() == Node_Leave || node_kind() == Node_Load || node_kind() == Node_Store || node_kind() == Node_Slot)
        //return false;

    //if (empty() || isa<Lambda>())
        //return true;

    //for (size_t i = 0, e = size(); i != e; ++i)
        //if (!op(i)->is_const())
            //return false;

    //return true;
//}

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

bool Def::is_primlit(int val) const {
    if (const PrimLit* lit = this->isa<PrimLit>()) {
        Box box = lit->box();
        switch (lit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return box.get_##T() == T(val);
#include "anydsl2/tables/primtypetable.h"
        }
    }
    return false;
}

bool Def::is_minus_zero() const {
    if (const PrimLit* lit = this->isa<PrimLit>()) {
        Box box = lit->box();
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
    Array<Use> uses = copy_uses();
    for_all (use, uses) {
        if (Lambda* lambda = use->isa_lambda())
            lambda->update_op(use.index(), with);
        else
            world().release(use->as<PrimOp>())->update(use.index(), with);
    }

    for_all (use, uses) {
        if (PrimOp* oprimop = (PrimOp*) use->isa<PrimOp>()) {
            Array<const Def*> ops(oprimop->ops());
            ops[use.index()] = with;
            size_t old_gid = world().gid();
            const Def* ndef = world().rebuild(oprimop, ops);

            if (oprimop->kind() == ndef->kind()) {
                assert(oprimop->size() == ndef->size());

                for (size_t i = 0, e = oprimop->size(); i != e; ++i) {
                    if (i != use.index() && oprimop->op(i) != ndef->op(i))
                        goto recurse;
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
            // update trackers to point to new defintion ndef
            for_all (tracker, oprimop->trackers())
                *tracker = ndef;

            oprimop->replace(ndef);
            oprimop->unset_ops();
            delete oprimop;
        }
    }
}

World& Def::world() const { return type_->world(); }
const Def* Def::op_via_lit(const Def* def) const { return op(def->primlit_value<size_t>()); }
Lambda* Def::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* Def::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int Def::order() const { return type()->order(); }
bool Def::is_generic() const { return type()->is_generic(); }
void Def::dump() const { dump(false); }

std::ostream& operator << (std::ostream& o, const anydsl2::Def* def) {
    Printer p(o, false);
    def->vdump(p);
    return p.o;
}

//------------------------------------------------------------------------------

Param::Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name)
    : Def(gid, Node_Param, 0, type, false, name)
    , lambda_(lambda)
    , index_(index)
    , representative_(0)
{}

Peeks Param::peek() const {
    size_t x = index();
    Lambda* l = lambda();
    Lambdas preds = l->direct_preds();
    Peeks result(preds.size());
    for_all2 (&res, result, pred, preds)
        res = Peek(pred->arg(x), pred);

    return result;
}

const Def* Param::representative() const {
    // TODO path compression
    for (const Param* param = this;;) {
        if (param->representative_) {
            if (param->representative_->isa<Param>())
                param = param->representative_->as<Param>();
            else {
                return param->representative_;
            }
        } else
            return param;
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
