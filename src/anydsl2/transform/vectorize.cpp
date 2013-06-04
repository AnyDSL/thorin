#include "anydsl2/transform/vectorize.h"

#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class Vectorizer {
public:

    Vectorizer(const Scope& scope, size_t length)
        : scope(scope)
        , pass(world().new_pass())
        , length(length)
    {}

    Lambda* vectorize();
    void create_conditions(Lambda* lambda);

    World& world() { return scope.world(); }
    const Def*& get_cond(Lambda* lambda) const { return (const Def*&) lambda->ptr; }

    const Scope& scope;
    size_t pass;
    const size_t length;
};

Lambda* Vectorizer::vectorize() {
    for_all (entry, scope.entries())
        get_cond(entry) = world().literal(true);

    for_all (lambda, scope.body())
        create_conditions(lambda);

    for_all (lambda, scope.rpo()) {
        lambda->dump_head();
        std::cout << "cond: ";
        get_cond(lambda)->dump();
    }

    return scope[0];
}

void Vectorizer::create_conditions(Lambda* lambda) {
    const Def*& cond = get_cond(lambda);

    Lambda* dom = scope.domtree().idom(lambda);
    if (scope.postdomtree().idom(dom) == lambda)
        cond = get_cond(dom);
    else {
        cond = world().literal(false);

        for_all (pred, scope.preds(lambda)) {
            const Def* pred_cond = get_cond(pred);

            if (Lambda* to = pred->to()->isa_lambda()) {
                assert(scope.num_succs(pred) == 1 && lambda == to);
            } else {
                const Select* select = pred->to()->as<Select>(); // conditional branch
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

const Type* vectorize(const Type* type, size_t length) {
    assert(!type->isa<VectorType>() || type->length() == 1);
    World& world = type->world();

    if (const PrimType* primtype = type->isa<PrimType>())
        return world.type(primtype->primtype_kind(), length);

    if (const Ptr* ptr = type->isa<Ptr>())
        return world.ptr(ptr->referenced_type(), length);

    Array<const Type*> new_elems(type->size());
    for_all2 (&new_elem, new_elems, elem, type->elems())
        new_elem = vectorize(elem, length);

    return world.rebuild(type, new_elems);
}

const Def* vectorize(const Def* cond, const Def* def) {
    World& world = cond->world();

    if (const PrimOp* primop = def->isa<PrimOp>()) {
        size_t size = primop->size();

        Array<const Def*> nops(primop->size());
        size_t i = 0;

        if (primop->isa<VectorOp>())
            nops[i++] = cond;

        for (; i != size; ++i)
            nops[i] = vectorize(cond, primop->op(i));

        return world.rebuild(primop, nops, vectorize(primop->type(), cond->length()));
    }

    return 0;
}

Lambda* vectorize(Scope& scope, size_t length) { return Vectorizer(scope, length).vectorize(); }

}
