#include "anydsl2/def.h"

#include <algorithm>
#include <stack>
#include <sstream>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/be/air.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

void DefNode::set_op(size_t i, const DefNode* def) {
    assert(!op(i) && "already set");
    set(i, def);
    if (isa<PrimOp>()) is_const_ &= def->is_const();
    auto p = def->uses_.insert(Use(i, this));
    assert(p.second && "already in use set");
}

void DefNode::unset_op(size_t i) {
    assert(op(i) && "must be set");
    unregister_use(i);
    set(i, nullptr);
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

Array<Use> DefNode::copy_uses() const {
    Array<Use> result(uses().size());
    std::copy(uses().begin(), uses().end(), result.begin());
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

void DefNode::replace(const DefNode* with) const {
#if 0
    std::unordered_set<const DefNode*> visited;

    std::stack<const DefNode*> stack;

    const DefNode* cur = this;
    do {
        if (auto lambda = cur->isa_lambda()) {
            assert(false && "todo");
        } else if (auto param = cur->isa<Param>()) {
            param
        } else {
            if (auto oprimop = cur->isa<PrimOp>()) {
            Array<const DefNode*> ops(oprimop->ops());
            for (auto index : use.indices())
                ops[index] = with;
            size_t old_gid = world().gid();
            const DefNode* ndef = world().rebuild(oprimop, ops);

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
    } while (!stack.empty());



        auto def = stack.top();
        stack.pop();
    }

    std::queue<const DefNode*> queue;
    for (auto use : this->uses())
        queue.push(use);

    while (!queue.empty()) {
        auto def = queue.front();
        queue.pop();

    }
    }

#if 0
    // copy trackers to avoid internal modification
    const Trackers trackers = trackers_;
    for (auto tracker : trackers)
        *tracker = with;

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
            Array<const DefNode*> ops(oprimop->ops());
            for (auto index : use.indices())
                ops[index] = with;
            size_t old_gid = world().gid();
            const DefNode* ndef = world().rebuild(oprimop, ops);

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
#endif
#endif
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
const DefNode* DefNode::op_via_lit(const DefNode* def) const { return op(def->primlit_value<size_t>()); }
Lambda* DefNode::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* DefNode::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
const PrimOp* DefNode::is_non_const_primop() const { return is_const() ? nullptr : isa<PrimOp>(); }
int DefNode::order() const { return type()->order(); }
bool DefNode::is_generic() const { return type()->is_generic(); }
size_t DefNode::length() const { return type()->as<VectorType>()->length(); }

//------------------------------------------------------------------------------

Param::Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name)
    : DefNode(gid, Node_Param, 0, type, false, name)
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
