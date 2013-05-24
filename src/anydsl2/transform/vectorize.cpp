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
    void create_conditions(const Def* cond, Lambda* lambda);

    World& world() { return scope.world(); }
    const Def*& get_cond(Lambda* lambda) const { return (const Def*&) lambda->ptr; }

    const Scope& scope;
    size_t pass;
    const size_t length;
};

Lambda* Vectorizer::vectorize() {
    create_conditions(world().true_mask(length), scope[0]);

    for_all (lambda, scope.rpo()) {
        lambda->dump_head();
        std::cout << "cond: ";
        get_cond(lambda)->dump();
    }

    return scope[0];
}

void Vectorizer::create_conditions(const Def* new_cond, Lambda* lambda) {
    const Def*& cond = get_cond(lambda);
    cond = lambda->visit(pass) ? world().arithop_or(cond, new_cond) : new_cond;

    if (Lambda* to = lambda->to()->isa_lambda()) {
        assert(scope.num_succs(lambda) == 1);
        create_conditions(cond, to);
    } else if (const Select* select = lambda->to()->isa<Select>()) { // conditional branch
        assert(scope.num_succs(lambda) == 2);
        create_conditions(world().arithop_and(cond,                     select->cond() ), select->tval()->as_lambda());
        create_conditions(world().arithop_and(cond, world().arithop_not(select->cond())), select->fval()->as_lambda());
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
