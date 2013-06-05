#include "anydsl2/transform/vectorize.h"

#include <sstream>

#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/topo_sort.h"

namespace anydsl2 {

class Vectorizer {
public:

    Vectorizer(const Scope& scope, size_t length)
        : scope(scope)
        , pass(world().new_pass())
        , length(length)
    {}

    Lambda* vectorize();
    void infer_condition(Lambda* lambda);
    void param2select(const Param* param);
    const Type* vectorize_type(const Type* type, size_t length);
    void vectorize_primop(const Def* cond, const PrimOp* primop);
    const Def* vectorize(const Def* def, size_t length);

    World& world() { return scope.world(); }
    static const Def*& map_cond(Lambda* lambda) { return (const Def*&) lambda->ptr; }
    static const Def*& map(const Def* def) { return (const Def*&) def->ptr; }

    const Scope& scope;
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
    for_all2 (&new_elem, new_elems, elem, type->elems())
        new_elem = vectorize_type(elem, length);

    return world.rebuild(type, new_elems);
}

Lambda* Vectorizer::vectorize() {
    Lambda* entry = scope.entries()[0];
    std::ostringstream oss;
    oss << scope[0]->name << "_x" << length;
    Lambda* vlambda = world().lambda(vectorize_type(entry->pi(), length)->as<Pi>(), LambdaAttr(LambdaAttr::Extern), oss.str());
    map_cond(entry) = world().literal(true, length);

    for_all2 (param, entry->params(), vparam, vlambda->params()) {
        map(param) = vparam;
        vparam->name = param->name;
    }

    // for all other stuff in topological order
    Lambda* cur = entry;
    std::vector<const Def*> topo = topo_sort(scope);
    for_all (def, ArrayRef<const Def*>(topo).slice_back(entry->num_params() + 1)) {
        if (Lambda* lambda = def->isa_lambda())
            infer_condition(cur = lambda);
        else if (const Param* param = def->isa<Param>())
            param2select(param);
        else {
            const PrimOp* primop = def->as<PrimOp>();
            if (primop->isa<Select>() && primop->type()->isa<Pi>())
                continue; // ignore branch
            vectorize_primop(map_cond(cur), primop);
        }
    }

    Lambda* exit = scope.exits()[0];
    Array<const Def*> vops(exit->size());
    for_all2 (&vop, vops, op, exit->ops())
        vop = vectorize(op, length);
    vlambda->jump(vops.front(), vops.slice_back(1));

    return vlambda;
}

void Vectorizer::infer_condition(Lambda* lambda) {
    const Def*& cond = map_cond(lambda);

    Lambda* dom = scope.domtree().idom(lambda);
    if (scope.postdomtree().idom(dom) == lambda)
        cond = map_cond(dom);
    else {
        cond = world().literal(false, length);

        for_all (pred, scope.preds(lambda)) {
            const Def* pred_cond = map_cond(pred);

            if (const Select* select = pred->to()->isa<Select>()) { // conditional branch
                assert(scope.num_succs(pred) == 2);
                const Def* select_cond = vectorize(select->cond(), length);
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

struct PredLess : public std::binary_function<const Lambda*, const Lambda*, bool> {
    bool operator () (const Lambda* l1, const Lambda* l2) const { 
        return Vectorizer::map(l1)->non_const_depth() > Vectorizer::map(l2)->non_const_depth();
    }
};

void Vectorizer::param2select(const Param* param) {
    const Def* select = 0;
    Array<Lambda*> preds = scope.preds(param->lambda());
    // begin with pred with the most expensive condition (non_const_depth) - this keeps select chains simpler
    std::sort(preds.begin(), preds.end(), PredLess());

    for_all (pred, preds) {
        const Def* peek = vectorize(pred->arg(param->index()), length);
        select = select ? world().select(map_cond(pred), peek, select) : peek;
    }

    map(param) = select;
    select->name = param->name;
}

void Vectorizer::vectorize_primop(const Def* cond, const PrimOp* primop) {
    size_t size = primop->size();
    Array<const Def*> vops(size);
    size_t i = 0;
    bool is_vector_op = primop->isa<VectorOp>();

    if (is_vector_op)
        vops[i++] = cond;

    for (; i != size; ++i)
        vops[i] = vectorize(primop->op(i), is_vector_op ? length : 1);

    map(primop) = world().rebuild(primop, vops, vectorize_type(primop->type(), is_vector_op ? length : 1));
}

const Def* Vectorizer::vectorize(const Def* def, size_t length) {
    if (def->isa<Param>() || def->is_non_const_primop())
        return map(def);
    if (const PrimLit* primlit = def->isa<PrimLit>())
        return world().literal(primlit->primtype_kind(), primlit->value(), length);
    if (def->isa<Bottom>())
        return world().bottom(def->type(), length);
    if (def->isa<Any>())
        return world().any(def->type(), length);

    const PrimOp* primop = def->as<PrimOp>();
    Array<const Def*> vops(primop->size());
    for_all2 (&vop, vops, op, primop->ops())
        vop = vectorize(op, length);

    return world().rebuild(primop, vops, vectorize_type(primop->type(), length));
}

Lambda* vectorize(Scope& scope, size_t length) { return Vectorizer(scope, length).vectorize(); }

}
