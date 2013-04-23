#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
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
    void place_late(Lambda* lambda, const PrimOp* primop);
    Lambda*& late(const PrimOp* primop) const { return (Lambda*&) primop->ptr; }
    void init(const PrimOp* primop) const {
        primop->ptr = 0;
        primop->counter = 0;
        for_all (use, primop->uses()) {
            if (use->isa<PrimOp>())
                ++primop->counter;
        }
    }

    Places place_early();
    void down(Places& places, Lambda* lambda);
    void place_early(Places& places, Lambda* early, const Def* def);

    const Scope& scope;
    size_t pass;
};

void Placement::place_late() {
    for (size_t i = scope.size(); i-- != 0;)
        up(scope[i]);
}

void Placement::up(Lambda* lambda) {
    for_all (op, lambda->ops()) {
        if (const PrimOp* primop = op->is_non_const_primop()) {
            if (!primop->visit(pass))
                init(primop);

            place_late(lambda, primop);
        }
    }
}

void Placement::place_late(Lambda* lambda, const PrimOp* primop) {
    late(primop) = late(primop) ? scope.domtree().lca(lambda, late(primop)) : lambda;

    for_all (op, primop->ops()) {
        if (const PrimOp* cur = op->is_non_const_primop()) {
            if (!cur->visit(pass))      
                init(cur);              // init unseen primops
            if (cur->counter == 0)      
                continue;               // skip branches which have already been processed
            if (--cur->counter == 0)    
                place_late(lambda, cur);// decrement and visit branch if all users have been processed
            assert(cur->counter != size_t(-1));
        }
    }
}

Places Placement::place_early() {
    Places places(scope.size());

    for_all (lambda, scope.rpo())
        down(places, lambda);

    return places;
}

void Placement::down(Places& places, Lambda* lambda) {
    for_all (param, lambda->params()) 
        place_early(places, lambda, param);
}

void Placement::place_early(Places& places, Lambda* early, const Def* def) {
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

        if (use->counter == 0) {
            if (const PrimOp* primop = use->isa<PrimOp>()) {
                Lambda* best = late(primop);
                if (primop->isa<Slot>() || primop->isa<Enter>())
                    best = early;                 // place these guys always early
                else if (!primop->isa<Leave>()) { // place this guy always late
                    // all other guys are placed as late as possible but keep them out of loops please
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
                place_early(places, early, primop);
            }
        }
    }
}

Places place(const Scope& scope) { return Placement(scope).place(); }

} // namespace anydsl2
