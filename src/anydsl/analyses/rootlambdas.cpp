#include "anydsl/analyses/rootlambdas.h"

#include "anydsl/lambda.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

static inline LambdaSet* depends(const Lambda* lambda) { return (LambdaSet*) lambda->scratch.ptr; }

static void depends(const Def* def, LambdaSet* dep) {
    if (const Param* param = def->isa<Param>())
        dep->insert(param->lambda());
    else if (!def->isa<Lambda>()) {
        for_all (op, def->ops())
            depends(op, dep);
    }
}

LambdaSet find_root_lambdas(const World& world) {
    return find_root_lambdas(world.lambdas());
}

LambdaSet find_root_lambdas(const LambdaSet& lambdas) {
    std::queue<const Lambda*> queue;
    LambdaSet inqueue;

    for_all (lambda, lambdas) {
        LambdaSet* dep = new LambdaSet();

        for_all (op, lambda->ops())
            depends(op, dep);

        lambda->scratch.ptr = dep;
        queue.push(lambda);
        inqueue.insert(lambda);
    }

    while (!queue.empty()) {
        const Lambda* lambda = queue.front();
        queue.pop();
        inqueue.erase(lambda);
        LambdaSet* dep = depends(lambda);
        size_t old = dep->size();

        for_all (succ, lambda->succs()) {
            LambdaSet* succ_dep = depends(succ);

            for_all (d, *succ_dep) {
                if (d != succ)
                    dep->insert(d);
            }
        }

        if (dep->size() != old) {
            for_all (pred, lambda->preds()) {
                if (inqueue.find(pred) == inqueue.end()) {
                    inqueue.insert(pred);
                    queue.push(pred);
                }
            }
        }
    }

    LambdaSet roots;

    for_all (lambda, lambdas) {
        LambdaSet* dep = depends(lambda);

        if (dep->size() == 1 && lambda == *dep->begin())
            roots.insert(lambda);

        delete dep;
    }

    return roots;
}

} // namespace anydsl
