#include "anydsl2/analyses/rootlambdas.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

static void find_user(const size_t pass, Lambda* entry, const Def* def);
static void up(const size_t pass, Lambda* entry, Lambda* lambda);
static void jump_to_param_users(const size_t pass, Lambda* entry, Lambda* lambda);

std::vector<Lambda*> find_root_lambdas(World& world) { 
    std::vector<Lambda*> result;
    size_t pass = world.new_pass();

    for_all (lambda, world.lambdas()) {
        if (!lambda->is_visited(pass))
            jump_to_param_users(pass, lambda, lambda);
    }

    for_all (lambda, world.lambdas())
        if (!lambda->is_visited(pass))
            result.push_back(lambda);

    return result;
}

static void jump_to_param_users(const size_t pass, Lambda* entry, Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(pass, entry, param);
}

static void find_user(const size_t pass, Lambda* entry, const Def* def) {
    if (Lambda* lambda = def->isa_lambda())
        up(pass, entry, lambda);
    else {
        if (def->visit(pass))
            return;

        for_all (use, def->uses())
            find_user(pass, entry, use);
    }
}

static void up(const size_t pass, Lambda* entry, Lambda* lambda) {
    if (lambda == entry || lambda->visit(pass))
        return;

    jump_to_param_users(pass, entry, lambda);

    for_all (pred, lambda->preds())
        up(pass, entry, pred);
}

} // namespace anydsl2
