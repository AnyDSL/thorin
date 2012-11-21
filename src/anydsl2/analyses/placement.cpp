#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class EarlyPlacement {
public:

    EarlyPlacement(const Scope& scope)
        : scope(scope)
        , pass(world().new_pass())
    {}

    World& world() const { return scope.world(); }
    Places place();
    void visit(Schedule& schedule, Lambda* lambda);
    void place(Schedule& schedule, const Def* def);
    void mark(const Def* def) { def->visit(pass); }
    bool defined(const Def* def) { return def->is_const() || def->is_visited(pass); }

private:

    const Scope& scope;
    size_t pass;
};

Places EarlyPlacement::place() {
    Places places(scope.size());

    for_all (lambda, scope.rpo())
        visit(places[lambda->sid()], lambda);

    return places;
}

void EarlyPlacement::visit(Schedule& schedule, Lambda* lambda) {
    for_all (param, lambda->params()) mark(param);
    for_all (param, lambda->params()) place(schedule, param);
}

void EarlyPlacement::place(Schedule& schedule, const Def* def) {
    assert(defined(def));

    for_all (use, def->uses()) {
        const Def* udef = use.def();

        if (udef->isa<Param>() || udef->isa<Lambda>())
            continue; // do not descent into lambdas -- it is handled by the RPO run

        for_all (op, udef->ops()) {
            if (!defined(op))
                goto outer_loop;
        }
        mark(udef);
        schedule.push_back(udef->as<PrimOp>());
        place(schedule, udef);
outer_loop: ;
    }
}

Places place_early(const Scope& scope) {
    EarlyPlacement placer(scope);
    return placer.place();
}

//------------------------------------------------------------------------------

#if 0
class LatePlacement {
public:

    LatePlacement(const Scope& scope)
        : scope(scope)
        , pass(world().new_pass())
    {}

    World& world() const { return scope.world(); }
    Places place();
    void visit(Schedule& schedule, Lambda* lambda);
    void place(Schedule& schedule, const Def* def);
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
    size_t pass;
};

Places LatePlacement::place() {
    size_t size = scope.size();
    Places places(size);

    for (size_t i = size; i-- != 0;) {
        Lambda* lambda = scope.rpo(i);
        visit(places[lambda->sid()], lambda);
    }

    return places;
}

void LatePlacement::visit(Schedule& schedule, Lambda* lambda) {
    for_all (op, lambda->ops()) inc(op);
    for_all (op, lambda->ops()) place(schedule, op);
}

void LatePlacement::place(Schedule& schedule, const Def* def) {
    if (def->isa<Param>() || def->isa<Lambda>())
        return; // do not descent into lambdas -- it is handled by the RPO run

    assert(all_uses_placed(def));
    for_all (op, def->ops()) {
        for_all (op, udef->ops()) {
            if (!defined(op))
                goto outer_loop;
        }
        mark(udef);
        schedule.push_back(udef->as<PrimOp>());
        place(schedule, udef);
outer_loop: ;
    }
}

Places place_late(const Scope& scope) {
    LatePlacement placer(scope);
    return placer.place();
}

#endif

} // namespace anydsl2
