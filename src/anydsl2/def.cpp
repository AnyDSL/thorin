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

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(std::find(def->uses_.begin(), def->uses_.end(), Use(i, this)) == def->uses_.end() && "already in use set");
    def->uses_.push_back(Use(i, this));
    set(i, def);
}

void Def::unset_op(size_t i) {
    assert(op(i) && "must be set");
    unregister_use(i);
    set(i, 0);
}

void Def::unregister_use(size_t i) const {
    const Def* def = op(i);
    Uses::iterator it = std::find(def->uses_.begin(), def->uses_.end(), Use(i, this));
    assert(it != def->uses_.end() && "must be in use set");
    def->uses_.erase(it);
}

bool Def::is_const() const {
    if (isa<Param>())
        return false;

    if (empty() || isa<Lambda>())
        return true;

    for (size_t i = 0, e = size(); i != e; ++i)
        if (!op(i)->is_const())
            return false;

    return true;
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

World& Def::world() const { return type_->world(); }
const Def* Def::op_via_lit(const Def* def) const { return op(def->primlit_value<size_t>()); }
Lambda* Def::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* Def::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int Def::order() const { return type()->order(); }
//void Def::replace(const Def* with) const { world().replace(this, with); }

std::ostream& operator << (std::ostream& o, const anydsl2::Def* def) {
    Printer p(o, false);
    def->vdump(p);
    return p.o;
}

//------------------------------------------------------------------------------

Param::Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name)
    : Def(gid, Node_Param, 0, type, name)
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
