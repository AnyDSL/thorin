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

class Placement {
public:

    Placement(const Scope& scope)
        : scope(scope)
        , pass(scope.world().new_pass())
    {}

    Places place() { 
        place_late(); 
        pass = scope.world().new_pass();
        return place_early(); 
    }

private:

    void place_late();
    void up(Lambda* lambda);
    void place_late(Lambda* lambda, const Def* def);
    bool is_visited(const PrimOp* primop) { return primop->is_visited(pass); }
    Lambda*& late(const PrimOp* primop) const { return (Lambda*&) primop->ptr; }

    Places place_early();
    void down(Places& places, Lambda* lambda);
    void place_early(Places& places, Lambda* early, const Def* def);
    void mark(const Def* def) { def->visit(pass); }
    bool defined(const Def* def) { return def->is_const() || def->is_visited(pass); }

    const Scope& scope;
    size_t pass;
};

void Placement::place_late() {
    for (size_t i = scope.size(); i-- != 0;)
        up(scope.rpo(i));
}

void Placement::up(Lambda* lambda) {
    for_all (op, lambda->ops()) 
        place_late(lambda, op);
}

void Placement::place_late(Lambda* lambda, const Def* def) {
    if (def->isa<Param>() || def->is_const())
        return;

    const PrimOp* primop = def->as<PrimOp>();

    if (is_visited(primop))
        late(primop) = scope.domtree().lca(late(primop), lambda);
    else {
        primop->visit(pass);
        late(primop) = lambda;
    }

    for_all (op, primop->ops())
        place_late(lambda, op);
}

Places Placement::place_early() {
    Places places(scope.size());

    for_all (lambda, scope.rpo())
        down(places, lambda);

    return places;
}

void Placement::down(Places& places, Lambda* lambda) {
    for_all (param, lambda->params()) mark(param);
    for_all (param, lambda->params()) place_early(places, lambda, param);
}

void Placement::place_early(Places& places, Lambda* early, const Def* def) {
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
            for (Lambda* i = best; i != early; i = scope.domtree().idom(i)) {
                int cur_depth = scope.loopinfo().depth(i);
                if (cur_depth < depth) {
                    best = i;
                    depth = cur_depth;
                }
            }
            places[best->sid()].push_back(primop);
            place_early(places, early, primop);
        }
outer_loop:;
    }
}

Places place(const Scope& scope) { return Placement(scope).place(); }

} // namespace anydsl2
