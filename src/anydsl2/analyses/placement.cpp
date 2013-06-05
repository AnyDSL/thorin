#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/topo_sort.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class Placement {
public:

    Placement(const Scope& scope)
        : scope(scope)
        , topo(topo_sort(scope))
    {}

    Places place() { 
        pass = scope.world().new_pass();
        place_late(); 
        return place_early(); 
    }

private:

    void place_late();
    void place_late(Lambda* lambda, const Def* def);
    Lambda*& get_late(const PrimOp* primop) const { return (Lambda*&) primop->ptr; }

    Places place_early();
    void place_early(Places& places, Lambda* early, const Def* def);

    const Scope& scope;
    std::vector<const Def*> topo;
    size_t pass;
};

void Placement::place_late() {
    for (size_t i = scope.size(); i-- != 0;)
        place_late(scope[i], scope[i]);
}

void Placement::place_late(Lambda* lambda, const Def* def) {
    for_all (op, def->ops()) {
        if (const PrimOp* primop = op->is_non_const_primop()) {
            if (!primop->visit(pass)) {     // init unseen primops
                primop->ptr = 0;
                primop->counter = 0;
                for_all (use, primop->uses()) {
                    if (use->isa<PrimOp>() || use->isa<Lambda>())
                        ++primop->counter;
                }
            }

            Lambda*& late = get_late(primop);
            late = late ? scope.domtree().lca(lambda, late) : lambda;

            if (--primop->counter == 0)     // only visit once when counter == 0
                place_late(late, primop);   // decrement and visit branch if all users have been processed
            assert(primop->counter != size_t(-1));
        }
    }
}

Places Placement::place_early() {
    Places places(scope.size());
    Lambda* early = 0;

    for_all (def, topo) {
        if (Lambda* lambda = def->isa_lambda())
            early = lambda;
        else if (const PrimOp* primop = def->isa<PrimOp>()) {
            if (!primop->is_visited(pass))
                continue;                           // primop is dead
            Lambda* best = get_late(primop);
            if (primop->isa<Slot>() || primop->isa<Enter>())
                best = early;                   // place these guys always early
            else if (!primop->isa<Leave>()) {   // place this guy always late
                // all other guys are placed as late as possible but keep them out of loops, please
                int depth = std::numeric_limits<int>::max();
                for (Lambda* i = best; i != early; i = scope.domtree().idom(i)) {
                    int cur_depth = scope.loopinfo().depth(i);
                    if (cur_depth < depth) {
                        best = i;
                        depth = cur_depth;
                    }
                }
            }
            places[best->sid()].push_back(primop);
        }
    }

    return places;
}

Places place(const Scope& scope) { return Placement(scope).place(); }

} // namespace anydsl2
