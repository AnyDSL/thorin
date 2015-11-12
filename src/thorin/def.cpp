#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
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
        assert_unused(res == 1);
        n->representative_ = target;
        const auto& p = target->representatives_of_.insert(n);
        assert_unused(p.second);
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
    const auto& p = node->uses_.emplace(i, this);
    assert_unused(p.second);
}

void DefNode::unregister_uses() const {
    for (size_t i = 0, e = size(); i != e; ++i)
        unregister_use(i);
}

void DefNode::unregister_use(size_t i) const {
    auto def = ops_[i].node();
    assert(def->uses_.count(Use(i, this)) == 1);
    def->uses_.erase(Use(i, this));
}

void DefNode::unlink_representative() const {
    if (is_proxy()) {
        auto num = this->representative_->representatives_of_.erase(this);
        assert_unused(num == 1);
    }
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
        auto cur = stack.top();
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
        const auto& p = with->representatives_of_.insert(this);
        assert_unused(p.second);

        std::queue<const DefNode*> queue;
        queue.push(this);
        while (!queue.empty()) {
            auto def = pop(queue);
            for (auto use : def->uses_) {
                if (auto uprimop = use.def().node()->isa<PrimOp>()) {
                    if (!uprimop->is_outdated()) {
                        uprimop->is_outdated_ = true;
                        queue.push(uprimop);
                    }
                }
            }
        }
    }
}

void DefNode::dump() const {
    auto primop = this->isa<PrimOp>();
    if (primop && !primop->is_const())
        primop->stream_assignment(std::cout);
    else {
        std::cout << this;
        std::cout << std::endl;
    }
}

World& DefNode::world() const { return type()->world(); }
Def DefNode::op(Def def) const { return op(def->primlit_value<size_t>()); }
Lambda* DefNode::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* DefNode::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int DefNode::order() const { return type()->order(); }
size_t DefNode::length() const { return type().as<VectorType>()->length(); }
std::ostream& DefNode::stream(std::ostream& out) const { return out << unique_name(); }

//------------------------------------------------------------------------------

}
