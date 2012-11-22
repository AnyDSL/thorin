#include "anydsl2/analyses/placement.h"

#include <queue>
#include <boost/unordered_map.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/loopforest.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

typedef boost::unordered_map<const PrimOp*, Lambda*> PrimOp2Lambda;

//------------------------------------------------------------------------------

class LatePlacement {
public:

    LatePlacement(const Scope& scope, const DomTree& domtree)
        : scope(scope)
        , domtree(domtree)
        , pass(world().new_pass())
    {}

    World& world() const { return scope.world(); }
    PrimOp2Lambda place();
    void visit(Lambda* lambda);
    void place(Lambda* lambda, const Def* def);
    void inc(const Def* def) { 
        if (def->visit(pass))
            ++def->counter;
        else
            def->counter = 1;
    }
    size_t num_placed(const Def* def) { return def->is_visited(pass) ? def->counter : 0; }
    bool all_uses_placed(const Def* def) { return num_placed(def) == def->num_uses(); }

private:

    const Scope& scope;
    const DomTree& domtree;
    size_t pass;
    PrimOp2Lambda primop2lambda;
};

PrimOp2Lambda LatePlacement::place() {
    size_t size = scope.size();
    for (size_t i = size; i-- != 0;)
        visit(scope.rpo(i));

    return primop2lambda;
}

void LatePlacement::visit(Lambda* lambda) {
    for_all (op, lambda->ops()) inc(op);
    for_all (op, lambda->ops()) place(lambda, op);
}

void LatePlacement::place(Lambda* lambda, const Def* def) {
    if (def->isa<Param>() || def->isa<Lambda>())
        return;
    const PrimOp* primop = def->as<PrimOp>();
    PrimOp2Lambda::const_iterator i = primop2lambda.find(primop);
    Lambda* lca = i == primop2lambda.end() ? lambda : domtree.lca(i->second, lambda);
    primop2lambda[primop] = lca;

    if (all_uses_placed(primop)) {
        for_all (op, primop->ops()) {
            inc(op);
            place(lambda, op);
        }
    }
}

static PrimOp2Lambda place_late(const Scope& scope, const DomTree& domtree) {
    LatePlacement placer(scope, domtree);
    return placer.place();
}

//------------------------------------------------------------------------------

class Placement {
public:

    Placement(const Scope& scope)
        : domtree(scope)
        , late(place_late(scope, domtree))
        , loopinfo(scope)
        , pass(world().new_pass())
    {}

    const Scope& scope() const { return loopinfo.scope(); }
    World& world() const { return scope().world(); }
    Places place();
    void visit(Places& places, Lambda* lambda);
    void place(Places& places, Lambda* early, const Def* def);
    void mark(const Def* def) { def->visit(pass); }
    bool defined(const Def* def) { return def->is_const() || def->is_visited(pass); }

private:

    DomTree domtree;
    PrimOp2Lambda late;
    LoopInfo loopinfo;
    size_t pass;
};

Places Placement::place() {
    Places places(scope().size());

    for_all (lambda, scope().rpo())
        visit(places, lambda);

    return places;
}

void Placement::visit(Places& places, Lambda* lambda) {
    for_all (param, lambda->params()) mark(param);
    for_all (param, lambda->params()) place(places, lambda, param);
}

void Placement::place(Places& places, Lambda* early, const Def* def) {
    assert(defined(def));

    for_all (use, def->uses()) {
        const Def* udef = use.def();
        if (udef->isa<Param>() || udef->isa<Lambda>())
            continue; // do not descent into lambdas -- it is handled by the RPO run
        const PrimOp* primop = udef->as<PrimOp>();

        for_all (op, primop->ops()) {
            if (!defined(op))
                goto outer_loop;
        }
        {
            mark(primop);
            Lambda* best = late[primop];
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
outer_loop: ;
    }
}

Places place(const Scope& scope) {
    Placement placer(scope);
    return placer.place();
}

//------------------------------------------------------------------------------

} // namespace anydsl2
