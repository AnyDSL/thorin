#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

//------------------------------------------------------------------------------

const DefNode* Def::deref() const {
    if (node_ == nullptr) return nullptr;

    auto target = node_;
    for (; target->is_proxy(); target = target->representative_)
        assert(target != nullptr);

    // path compression
    auto n = node_;
    while (n->representative_ != target) {
        auto representative = n->representative_;
        auto res = representative->representatives_of_.erase(n);
        assert(res == 1);
        n->representative_ = target;
        target->representatives_of_.insert(n);
        n = representative;
    }

    return node_ = target;
}

void DefNode::set_op(size_t i, Def def) {
    assert(!op(i) && "already set");
    auto node = *def;
    ops_[i] = node;
    if (isa<PrimOp>())
        is_const_ &= node->is_const();
    assert(def->uses_.count(Use(i, this)) == 0);
    node->uses_.emplace(i, this);
}

void DefNode::unregister_use(size_t i) const {
    auto def = op(i).node();
    assert(def->uses_.count(Use(i, this)) == 1);
    def->uses_.erase(Use(i, this));
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
#define THORIN_UF_TYPE(T) case PrimType_##T: return box.get_##T() == T(val);
#include "thorin/tables/primtypetable.h"
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
#define THORIN_JUST_U_TYPE(T) case PrimType_##T: return box.get_##T() == T(0);
#include "thorin/tables/primtypetable.h"
            case PrimType_f32: return box.get_f32() == -0.f;
            case PrimType_f64: return box.get_f64() == -0.0;
        }
    }
    return false;
}

void DefNode::replace(Def with) const {
    assert(type() == with->type());
    if (this == *with) return;
    assert(!is_proxy() && !is_const());
    assert(!isa<Param>() || !as<Param>()->lambda()->attribute().is(Lambda::Extern | Lambda::Intrinsic));
    this->representative_ = with;
    auto p = with->representatives_of_.insert(this);
    assert(p.second);
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
    Peeks peeks;
    for (auto use : lambda()->uses()) {
        if (auto pred = use->isa_lambda()) {
            if (use.index() == 0)
                peeks.emplace_back(pred->arg(index()), pred);
        } 
        else if (auto evalop = use->isa<EvalOp>()) {
            for (auto use : evalop->uses()) {
                if (auto pred = use->isa_lambda())
                    if (use.index() == 0)
                        peeks.emplace_back(pred->arg(index()), pred);
            }
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

} // namespace thorin
