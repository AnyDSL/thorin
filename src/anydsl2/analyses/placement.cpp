#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

std::vector<const Def*> visit_early(const Scope& scope) {
    std::vector<const Def*> result;
    std::queue<const Def*> queue;
    const size_t pass = scope.world().new_pass();

    for_all (lambda, scope.rpo()) {
        result.push_back(lambda);

        for_all (param, lambda->params())
            queue.push(param);

        while (!queue.empty()) {
            const Def* def = queue.front();
            result.push_back(def);
            queue.pop();

            for_all (use, def->uses()) {
                if (use->isa<Lambda>())
                    continue;
                if (use->visit(pass))
                    --use->counter;
                else {
                    use->counter = -1;
                    for_all (op, use->ops()) {
                        if (!op->is_const())
                            ++use->counter;
                    }
                }
                assert(use->counter != size_t(-1));

                if (use->counter == 0)
                    queue.push(use);
            }
        }
    }

    return result;
}

Lambda*& get_late(const PrimOp* primop) { return (Lambda*&) primop->ptr; }

std::vector<const Def*> visit_late(const Scope& scope) {
    std::vector<const Def*> result;
    Array< std::vector<const PrimOp*> > lambda2primop(scope.size());
    std::queue<const Def*> queue;
    const size_t pass = scope.world().new_pass();

    for (size_t i = scope.size(); i-- != 0;) {
        Lambda* cur = scope[i];

        queue.push(cur);

        while (!queue.empty()) {
            const Def* def = queue.front();
            queue.pop();

            for_all (op, def->ops()) {
                if (const PrimOp* primop = op->is_non_const_primop()) {
                    if (!primop->visit(pass)) {     // init unseen primops
                        primop->ptr = 0;
                        primop->counter = primop->num_uses();
                    }

                    Lambda*& late = get_late(primop);
                    late = late ? scope.domtree().lca(cur, late) : cur;

                    if (--primop->counter == 0) {   // only visit once when counter == 0
                        lambda2primop[late->sid()].push_back(primop);
                        queue.push(primop);
                    }

                    assert(primop->counter != size_t(-1));
                }
            }
        }

        for_all (primop, lambda2primop[cur->sid()])
            result.push_back(primop);

        for (size_t i = cur->num_params(); i-- != 0;)
            result.push_back(cur->param(i));

        result.push_back(cur);
    }

    std::reverse(result.begin(), result.end());
    return result;
}

class Placement {
public:

    Placement(const Scope& scope)
        : scope(scope)
        , topo_order(visit_early(scope))
    {}

    Places place() { 
        pass = scope.world().new_pass();
        place_late(); 
        return place_early(); 
    }

    void place_late();

private:

    void place_late(Lambda* lambda, const Def* def);
    Lambda*& get_late(const PrimOp* primop) const { return (Lambda*&) primop->ptr; }

    Places place_early();
    void place_early(Places& places, Lambda* early, const Def* def);

    const Scope& scope;
    std::vector<const Def*> topo_order;
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
                primop->counter = primop->num_uses();
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

    for_all (def, topo_order) {
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
