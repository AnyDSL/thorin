#include <algorithm>
#include <sstream>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

class Vectorizer {
public:
    Vectorizer(Scope& scope, size_t length)
        : scope(scope)
        , domtree(scope.domtree())
        , postdomtree(scope.postdomtree())
        , length(length)
    {}

    Lambda* vectorize();
    void infer_condition(Lambda* lambda);
    void param2select(const Param* param);
    Type vectorize_type(Type, size_t length);
    void vectorize_primop(Def cond, const PrimOp* primop);
    Def vectorize(Def def, size_t length);

    World& world() { return scope.world(); }

    Scope& scope;
    const DomTree* domtree;
    const PostDomTree* postdomtree;
    Def2Def mapped;
    const size_t length;
};

Type Vectorizer::vectorize_type(Type type, size_t length) {
    assert(!type.isa<VectorType>() || type->length() == 1);
    World& world = type->world();

    if (auto primtype = type.isa<PrimType>())
        return world.type(primtype->primtype_kind(), length);

    if (auto ptr = type.isa<PtrType>())
        return world.ptr_type(ptr->referenced_type(), length);

    Array<Type> new_args(type->num_args());
    for (size_t i = 0, e = type->num_args(); i != e; ++i)
        new_args[i] = vectorize_type(type->arg(i), length);

    return world.rebuild(type, new_args);
}

Lambda* Vectorizer::vectorize() {
    std::ostringstream oss;
    auto entry = scope.entry();
    oss << entry->name << "_x" << length;
    auto vlambda = world().lambda(vectorize_type(entry->type(), length).as<FnType>(), oss.str());
    vlambda->make_external();
    mapped[entry] = *world().true_mask(length);

    for (size_t i = 0, e = entry->num_params(); i != e; ++i) {
        const Param* param = entry->param(i);
        const Param* vparam = vlambda->param(i);
        mapped[param] = vparam;
        vparam->name = param->name;
    }

    Schedule schedule = schedule_smart(scope);

    for (auto lambda : scope) {
        if (lambda != entry) {
            infer_condition(lambda);
            for (auto param : lambda->params())
                param2select(param);
        }

        for (auto primop : schedule[lambda]) {
            if (primop->isa<Select>() && primop->type().isa<FnType>())
                continue; // ignore branch
            vectorize_primop(mapped[lambda], primop);
        }
    }

    auto exit = scope.exit();
    Array<Def> vops(exit->size());
    for (size_t i = 0, e = exit->size(); i != e; ++i)
        vops[i] = vectorize(exit->op(i), length);
    vlambda->jump(vops.front(), vops.slice_from_begin(1));

    return vlambda;
}

void Vectorizer::infer_condition(Lambda* lambda) {
    const DefNode*& cond = mapped[lambda];

    Lambda* dom = domtree->idom(lambda);
    if (postdomtree->idom(dom) == lambda)
        cond = mapped[dom];
    else {
        cond = world().false_mask(length);

        for (auto pred : scope.preds(lambda)) {
            Def pred_cond = mapped[pred];

            if (const Select* select = pred->to()->isa<Select>()) { // conditional branch
                assert(scope.num_succs(pred) == 2);
                Def select_cond = vectorize(select->cond(), length);
                if (select->tval() == lambda)
                    pred_cond = world().arithop_and(pred_cond, select_cond);
                else {
                    assert(select->fval() == lambda);
                    pred_cond = world().arithop_and(pred_cond, world().arithop_not(select_cond));
                }
            }

            cond = world().arithop_or(cond, pred_cond);
        }
    }
}

static int non_const_depth(Def def) {
    if (def->is_const() || def->isa<Param>())
        return 0;

    auto primop = def->as<PrimOp>();
    int max = 0;
    for (auto op : primop->ops()) {
        int d = non_const_depth(op);
        max = d > max ? d : max;
    }

    return max + 1;
}

void Vectorizer::param2select(const Param* param) {
    Def select = nullptr;
    Array<Lambda*> preds = scope.preds(param->lambda()); // copy
    // begin with pred with the most expensive condition (non_const_depth) - this keeps select chains simpler
    std::stable_sort(preds.begin(), preds.end(), [&](const Lambda* l1, const Lambda* l2) {
        return non_const_depth(mapped[l1]) > non_const_depth(mapped[l2]);
    });

    for (auto pred : preds) {
        Def peek = vectorize(pred->arg(param->index()), length);
        select = select ? world().select(mapped[pred], peek, select) : peek;
    }

    mapped[param] = select;
    select->name = param->name;
}

void Vectorizer::vectorize_primop(Def cond, const PrimOp* primop) {
    size_t size = primop->size();
    Array<Def> vops(size);
    size_t i = 0;
    bool is_vector_op = primop->isa<VectorOp>() != nullptr;

    if (is_vector_op)
        vops[i++] = cond;

    for (; i != size; ++i)
        vops[i] = vectorize(primop->op(i), is_vector_op ? length : 1);

    mapped[primop] = world().rebuild(primop, vops, vectorize_type(primop->type(), is_vector_op ? length : 1));
}

Def Vectorizer::vectorize(Def def, size_t length) {
    if (def->isa<Param>() || (def->isa<PrimOp>() && !def->is_const()))
        return mapped[def];
    if (auto primlit = def->isa<PrimLit>())
        return world().literal(primlit->primtype_kind(), primlit->value(), length);
    if (def->isa<Bottom>())
        return world().bottom(def->type(), length);
    if (def->isa<Any>())
        return world().any(def->type(), length);

    const PrimOp* primop = def->as<PrimOp>();
    Array<Def> vops(primop->size());
    for (size_t i = 0, e = primop->size(); i != e; ++i)
        vops[i] = vectorize(primop->op(i), length);

    return world().rebuild(primop, vops, vectorize_type(primop->type(), length));
}

Lambda* vectorize(Scope& scope, size_t length) { return Vectorizer(scope, length).vectorize(); }

}
