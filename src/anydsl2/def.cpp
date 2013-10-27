#include "anydsl2/def.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/be/air.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

const DefNode* Def::deref() const {
    const DefNode* n = node_;
    for (; n != nullptr && n != n->representative_; n = n->representative_)
        assert(n != nullptr);

    return n;
}

void DefNode::set_op(size_t i, Def def) {
    assert(!op(i) && "already set");
    ops_[i] = def;
    if (isa<PrimOp>()) is_const_ &= def->is_const();
    auto p = def->uses_.emplace(i, this);
    assert(p.second && "already in use set");
}

void DefNode::unregister_use(size_t i) const { 
    auto def = op(i).node();
    auto res = def->uses_.erase(Use(i, this));
    assert(res == 1);
    if (def->is_proxy())
        def->representative_->representatives_of_.erase(def);
}

void DefNode::unset_op(size_t i) {
    assert(op(i) && "must be set");
    unregister_use(i);
    ops_[i] = nullptr;
}

void DefNode::unset_ops() {
    for (size_t i = 0, e = size(); i != e; ++i)
        unset_op(i);
}

std::string DefNode::unique_name() const {
    std::ostringstream oss;
    oss << name << '_' << gid();
    return oss.str();
}

std::vector<Use> DefNode::uses() const {
    std::vector<Use> result;
    std::vector<const DefNode*> stack;
    stack.push_back(this);

    while (!stack.empty()) {
        const DefNode* cur = stack.back();
        stack.pop_back();

        for (auto use : cur->uses_) {
            if (!use.def().node()->is_proxy())
                result.push_back(use);
        }

        for (auto of : cur->representatives_of_)
            stack.push_back(of);
    }

    return result;
}

bool DefNode::is_primlit(int val) const {
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

bool DefNode::is_minus_zero() const {
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

void DefNode::replace(Def with) const {
    assert(!is_proxy() && !is_const() && this != *with);
    assert(!isa<Param>() || !as<Param>()->lambda()->attribute().is(Lambda::Extern));
    this->representative_ = with;
    with->representatives_of_.insert(this);
}

int DefNode::non_const_depth() const {
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

void DefNode::dump() const { 
    auto primop = this->isa<PrimOp>();
    if (primop && !primop->is_const())
        emit_assignment(primop);
    else {
        emit_def(this); 
        std::cout << std::endl;
    }
}

World& DefNode::world() const { return type()->world(); }
Def DefNode::op_via_lit(Def def) const { return op(def->primlit_value<size_t>()); }
Lambda* DefNode::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* DefNode::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
const PrimOp* DefNode::is_non_const_primop() const { return is_const() ? nullptr : isa<PrimOp>(); }
int DefNode::order() const { return type()->order(); }
bool DefNode::is_generic() const { return type()->is_generic(); }
size_t DefNode::length() const { return type()->as<VectorType>()->length(); }

//------------------------------------------------------------------------------

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
