#include <algorithm>
#include <sstream>

#include "thorin/literal.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

class Vectorizer {
public:
    Vectorizer(const Scope& scope, size_t length)
        : scope(scope)
        , domtree(DomTree(scope))
        , postdomtree(PostDomTree(scope))
        , length(length)
    {}

    Lambda* vectorize();
    void infer_condition(Lambda* lambda);
    void param2select(const Param* param);
    const Type* vectorize_type(const Type* type, size_t length);
    void vectorize_primop(Def cond, const PrimOp* primop);
    Def vectorize(Def def, size_t length);

    World& world() { return scope.world(); }

    const Scope& scope;
    const DomTree domtree;
    const PostDomTree postdomtree;
    Def2Def mapped;
    size_t pass;
    const size_t length;
};

const Type* Vectorizer::vectorize_type(const Type* type, size_t length) {
    assert(!type->isa<VectorType>() || type->length() == 1);
    World& world = type->world();

    if (const PrimType* primtype = type->isa<PrimType>())
        return world.type(primtype->primtype_kind(), length);

    if (const Ptr* ptr = type->isa<Ptr>())
        return world.ptr(ptr->referenced_type(), length);

    Array<const Type*> new_elems(type->size());
    for (size_t i = 0, e = type->size(); i != e; ++i)
        new_elems[i] = vectorize_type(type->elem(i), length);

    return world.rebuild(type, new_elems);
}

Lambda* Vectorizer::vectorize() {
    Lambda* entry = scope.entries()[0];
    std::ostringstream oss;
    oss << scope[0]->name << "_x" << length;
    Lambda* vlambda = world().lambda(vectorize_type(entry->pi(), length)->as<Pi>(), Lambda::Attribute(Lambda::Extern), oss.str());
    mapped[entry] = *world().true_mask(length);

    for (size_t i = 0, e = entry->num_params(); i != e; ++i) {
        const Param* param = entry->param(i);
        const Param* vparam = vlambda->param(i);
        mapped[param] = vparam;
        vparam->name = param->name;
    }

    Schedule schedule = schedule_early(scope);

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda = scope[i];

        if (i != 0) {
            infer_condition(lambda);
            for (auto param : lambda->params())
                param2select(param);
        }

        for (auto primop : schedule[lambda]) {
            if (primop->isa<Select>() && primop->type()->isa<Pi>())
                continue; // ignore branch
            vectorize_primop(mapped[lambda], primop);
        }
    }

    Lambda* exit = scope.exits()[0];
    Array<Def> vops(exit->size());
    for (size_t i = 0, e = exit->size(); i != e; ++i)
        vops[i] = vectorize(exit->op(i), length);
    vlambda->jump(vops.front(), vops.slice_from_begin(1));

    return vlambda;
}

void Vectorizer::infer_condition(Lambda* lambda) {
    const DefNode*& cond = mapped[lambda];

    Lambda* dom = domtree.idom(lambda);
    if (postdomtree.idom(dom) == lambda)
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

void Vectorizer::param2select(const Param* param) {
    Def select = nullptr;
    auto preds = scope.preds(param->lambda()); // copy
    // begin with pred with the most expensive condition (non_const_depth) - this keeps select chains simpler
    std::sort(preds.begin(), preds.end(), [&](const Lambda* l1, const Lambda* l2) {
        return mapped[l1]->non_const_depth() > mapped[l2]->non_const_depth(); 
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
    if (def->isa<Param>() || def->is_non_const_primop())
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
