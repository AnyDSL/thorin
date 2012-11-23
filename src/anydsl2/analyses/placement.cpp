#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/loopforest.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class LatePlacement {
public:

    LatePlacement(const Scope& scope, const DomTree& domtree)
        : scope(scope)
        , domtree(domtree)
        , pass(world().new_pass())
    {
        size_t size = scope.size();
        for (size_t i = size; i-- != 0;)
            up(scope.rpo(i));
    }

    World& world() const { return scope.world(); }
    void place();
    void up(Lambda* lambda);
    void place(Lambda* lambda, const Def* def);
    bool is_visited(const PrimOp* primop) { return primop->is_visited(pass); }
    Lambda*& late(const PrimOp* primop) const { return (Lambda*&) primop->ptr; }

private:

    const Scope& scope;
    const DomTree& domtree;
    size_t pass;
};

void LatePlacement::up(Lambda* lambda) {
    for_all (op, lambda->ops()) 
        place(lambda, op);
}

void LatePlacement::place(Lambda* lambda, const Def* def) {
    if (def->isa<Param>() || def->is_const())
        return;

    const PrimOp* primop = def->as<PrimOp>();
    late(primop) = is_visited(primop) ? domtree.lca(late(primop), lambda) : (primop->visit(pass), lambda);

    for_all (op, primop->ops())
        place(lambda, op);
}

//------------------------------------------------------------------------------

class Placement {
public:

    Placement(const Scope& scope)
        : domtree(scope)
        , late_placement(scope, domtree)
        , loopinfo(scope)
        , pass(world().new_pass())
    {}

    const Scope& scope() const { return loopinfo.scope(); }
    World& world() const { return scope().world(); }
    Places place();
    void down(Places& places, Lambda* lambda);
    void place(Places& places, Lambda* early, const Def* def);
    void mark(const Def* def) { def->visit(pass); }
    bool defined(const Def* def) { return def->is_const() || def->is_visited(pass); }
    Lambda* late(const PrimOp* primop) const { return (Lambda*) primop->ptr; }

private:

    DomTree domtree;
    LatePlacement late_placement;
    LoopInfo loopinfo;
    size_t pass;
};

Places Placement::place() {
    Places places(scope().size());

    for_all (lambda, scope().rpo())
        down(places, lambda);

    return places;
}

void Placement::down(Places& places, Lambda* lambda) {
    for_all (param, lambda->params()) mark(param);
    for_all (param, lambda->params()) place(places, lambda, param);
}

void Placement::place(Places& places, Lambda* early, const Def* def) {
    assert(defined(def));

    for_all (use, def->uses()) {
        const Def* udef = use.def();
        if (defined(udef))
            continue;
        const PrimOp* primop = udef->as<PrimOp>();

        for_all (op, primop->ops()) {
            if (!defined(op))
                goto outer_loop;
        }
        {
            mark(primop);
            Lambda* best = late(primop);
            int depth = std::numeric_limits<int>::max();
            for (Lambda* i = best; i != early; i = domtree.idom(i)) {
                int cur_depth = loopinfo.depth(i);
                if (cur_depth < depth) {
                    best = i;
                    depth = cur_depth;
                }
            }
            places[best->sid()].push_back(primop);
            place(places, early, primop);
        }
outer_loop:;
    }
}

Places place(const Scope& scope) {
    Placement placer(scope);
    return placer.place();
}

//------------------------------------------------------------------------------

} // namespace anydsl2
