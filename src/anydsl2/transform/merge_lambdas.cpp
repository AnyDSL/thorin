#include "anydsl2/transform/merge_lambdas.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

class Merger {
public:

    Merger(World& world)
        : world(world)
    {}

    void merge();

private:

    World& world;
};

void Merger::merge() {
    for_all (lambda, world.lambdas()) {
        if (lambda->num_uses() == 1 && !lambda->attr().is_extern() && lambda->num_params() == 0) {
            Use use = *lambda->uses().begin();
            if (use.index() == 0) {
                Lambda* ulambda = use.def()->as_lambda();
                ulambda->jump(lambda->to(), lambda->args());
                lambda->destroy_body();
            }
        }
    }
}

void merge_lambdas(World& world) { Merger(world).merge(); }

} // namespace anydsl2
