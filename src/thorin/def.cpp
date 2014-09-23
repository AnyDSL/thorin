#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

const DefNode* Def::deref() const {
    if (node_ == nullptr) return nullptr;

    auto target = node_;
    for (; target->is_proxy(); target = target->representative_)
        assert(target != nullptr);

    // path compression
    const DefNode* n = node_;
    while (n->representative_ != target) {
        auto representative = n->representative_;
        auto res = representative->representatives_of_.erase(n);
        assert(res == 1);
        n->representative_ = target;
        auto p = target->representatives_of_.insert(n);
        assert(p.second);
        n = representative;
    }

    return target;
}

void DefNode::set_op(size_t i, Def def) {
    assert(!op(i) && "already set");
    assert(def && "setting null pointer");
    auto node = *def;
    ops_[i] = node;
    assert(def->uses_.count(Use(i, this)) == 0);
    auto p = node->uses_.emplace(i, this);
    assert(p.second);
}

void DefNode::unregister_use(size_t i) const {
    auto def = ops_[i].node();
    assert(def->uses_.count(Use(i, this)) == 1);
    def->uses_.erase(Use(i, this));
}

void DefNode::unset_op(size_t i) {
    assert(ops_[i] && "must be set");
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

Def DefNode::refresh() const {
    if (up_to_date_ || isa<Param>() || isa<Lambda>())
        return this;

    auto oprimop = as<PrimOp>();
    Array<Def> ops(oprimop->size());
    for (size_t i = 0, e = oprimop->size(); i != e; ++i)
        ops[i] = oprimop->op(i)->refresh();

    auto nprimop = world().rebuild(oprimop, ops);
    //assert(nprimop != oprimop);
    //this->representative_ = nprimop;
    //auto p = nprimop->representatives_of_.insert(this);
    //assert(p.second);
    return nprimop;
}

Def DefNode::op(size_t i) const {
    assert(i < ops().size() && "index out of bounds");
    //assert(up_to_date_ && "retrieving operand of Def that is not up-to-date");

    auto op = ops_[i];
    if (op && !op->up_to_date_) {
        op = op->refresh();
        if (auto lambda = isa_lambda())
            lambda->update_op(i, op);
        ops_[i]->replace(op);
    }
    return op;
}

bool DefNode::is_const() const {
    if (isa<Param>()) return false;
    if (isa<PrimOp>()) {
        for (auto op : ops()) { // TODO slow because ops form a DAG not a tree
            if (!op->is_const())
                return false;
        }
    }

    return true; // lambdas are always const
}

std::vector<Use> DefNode::uses() const {
    std::vector<Use> result;
    std::stack<const DefNode*> stack;
    stack.push(this);

    while (!stack.empty()) {
        const DefNode* cur = stack.top();
        stack.pop();

        for (auto use : cur->uses_) {
            if (!use.def().node()->is_proxy())
                result.push_back(use);
        }

        for (auto of : cur->representatives_of_)
            stack.push(of);
    }

    return result;
}

bool DefNode::is_primlit(int val) const {
    if (auto lit = this->isa<PrimLit>()) {
        switch (lit->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit->value().get_##T() == T(val);
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
#define THORIN_I_TYPE(T, M) case PrimType_##T: return box.get_##T() == T(0);
#define THORIN_F_TYPE(T, M) case PrimType_##T: return box.get_##T() == T(-0.0);
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }
    }
    return false;
}

void DefNode::replace(Def with) const {
    assert(type() == with->type());
    if (this != *with) {
        assert(!is_proxy());
        this->representative_ = with;
        auto p = with->representatives_of_.insert(this);
        assert(p.second);

        std::queue<const DefNode*> queue;
        queue.push(this);
        while (!queue.empty()) {
            auto def = pop(queue);
            def->up_to_date_ = false;

            for (auto use : def->uses_) {
                if (use.def().node()->up_to_date_ && !use.def().node()->isa_lambda())
                    queue.push(use.def().node());
            }
        }
    }
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
Def DefNode::op(Def def) const { return op(def->primlit_value<size_t>()); }
Lambda* DefNode::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* DefNode::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int DefNode::order() const { return type()->order(); }
size_t DefNode::length() const { return type().as<VectorType>()->length(); }

//------------------------------------------------------------------------------

std::vector<Param::Peek> Param::peek() const {
    std::vector<Peek> peeks;
    for (auto use : lambda()->uses()) {
        if (auto pred = use->isa_lambda()) {
            if (use.index() == 0)
                peeks.emplace_back(pred->arg(index()), pred);
        } else if (auto evalop = use->isa<EvalOp>()) {
            for (auto use : evalop->uses()) {
                if (auto pred = use->isa_lambda()) {
                    if (use.index() == 0)
                        peeks.emplace_back(pred->arg(index()), pred);
                }
            }
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

}
