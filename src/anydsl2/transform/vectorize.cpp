#include "anydsl2/transform/vectorize.h"

#include <sstream>

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
    const Type* vectorize_type(const Type* type);
    const Def* vectorize_primop(const Def* cond, const PrimOp* primop);
    const Def* vectorize(const Def* def, size_t length);

    World& world() { return scope.world(); }
    const Def*& map_cond(Lambda* lambda) const { return (const Def*&) lambda->ptr; }
    const Def*& map(const Def* def) const { return (const Def*&) def->ptr; }

    const Scope& scope;
    size_t pass;
    const size_t length;
    Lambda* vlambda;
};

const Type* Vectorizer::vectorize_type(const Type* type) {
    assert(!type->isa<VectorType>() || type->length() == 1);
    World& world = type->world();

    if (const PrimType* primtype = type->isa<PrimType>())
        return world.type(primtype->primtype_kind(), length);

    if (const Ptr* ptr = type->isa<Ptr>())
        return world.ptr(ptr->referenced_type(), length);

    Array<const Type*> new_elems(type->size());
    for_all2 (&new_elem, new_elems, elem, type->elems())
        new_elem = vectorize_type(elem);

    return world.rebuild(type, new_elems);
}

Lambda* Vectorizer::vectorize() {
    Lambda* entry = scope.entries()[0];
    std::ostringstream oss;
    oss << scope[0]->name << "_x" << length;
    vlambda = world().lambda(vectorize_type(entry->pi())->as<Pi>(), oss.str());
    map_cond(entry) = world().literal(true, length);

    for_all2 (param, entry->params(), vparam, vlambda->params())
        map(param) = vparam;

    Lambda* cur = entry;
    // for all other stuff in topological order
    std::vector<const Def*> topo = topo_sort(scope);
    for_all (def, ArrayRef<const Def*>(topo).slice_back(entry->num_params() + 1)) {
        if (Lambda* lambda = def->isa_lambda())
            infer_condition(cur = lambda);
        else if (const Param* param = def->isa<Param>())
            param2select(param);
        else
            vectorize_primop(map_cond(cur), def->as<PrimOp>());
    }

    //Lambda* exit = scope.exits()[0];
    vlambda->dump_head();

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
                if (select->tval() == lambda)
                    pred_cond = world().arithop_and(pred_cond, select->cond());
                else {
                    assert(select->fval() == lambda);
                    pred_cond = world().arithop_and(pred_cond, world().arithop_not(select->cond()));
                }
            }

            cond = world().arithop_or(cond, pred_cond);
        }
    }
}

void Vectorizer::param2select(const Param* param) {
    const Def* select = 0;
    for_all (pred, scope.preds(param->lambda()))
        select = select ? world().select(map_cond(pred), vectorize(pred->arg(param->index()), length), select) : map_cond(pred);

    map(param) = select;
}

const Def* Vectorizer::vectorize_primop(const Def* cond, const PrimOp* primop) {
    size_t size = primop->size();
    Array<const Def*> nops(primop->size());
    size_t i = 0;

    if (primop->isa<VectorOp>())
        nops[i++] = cond;

    for (; i != size; ++i)
        nops[i] = vectorize(primop->op(i));

    return world().rebuild(primop, nops, vectorize_type(primop->type()));
}

const Def* Vectorizer::vectorize(const Def* def, size_t length) {
    return def;
}

Lambda* vectorize(Scope& scope, size_t length) { return Vectorizer(scope, length).vectorize(); }

}
