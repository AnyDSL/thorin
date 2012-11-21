#include "anydsl2/analyses/placement.h"

#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class Placer {
public:

    Placer(const Scope& scope)
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

Places Placer::place() {
    Places places(scope.size());

    for_all (lambda, scope.rpo())
        visit(places[lambda->sid()], lambda);

    return places;
}

void Placer::visit(Schedule& schedule, Lambda* lambda) {
    for_all (param, lambda->params()) mark(param);
    for_all (param, lambda->params()) place(schedule, param);
}

void Placer::place(Schedule& schedule, const Def* def) {
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

Places place(const Scope& scope) {
    Placer placer(scope);
    return placer.place();
}

} // namespace anydsl2
