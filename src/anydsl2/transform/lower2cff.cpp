#include <iostream>
#include <unordered_map>

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

class CFFLowering {
public:
    CFFLowering(World& world) {
        Array<Lambda*> top = top_level_lambdas(world);
        top_.insert(top.begin(), top.end());
    }

    void transform(Lambda* lambda);
    size_t process();

private:
    LambdaSet top_;
};

void CFFLowering::transform(Lambda* lambda) {
    Scope scope(lambda);
    std::unordered_map<Array<const Def*>, Lambda*> args2lambda;

    for (auto use : lambda->copy_uses()) {
        if (use.index() != 0 || !use->isa<Lambda>())
            continue;

        Lambda* ulambda = use->as_lambda();
        if (scope.contains(ulambda))
            continue;

        GenericMap generic_map;
        bool res = lambda->type()->infer_with(generic_map, ulambda->arg_pi());
        assert(res);
        
        size_t size = lambda->num_params();
        Array<size_t> indices(size);
        Array<const Def*> with(size);
        Array<const Def*> args(size);

        // don't drop the "return" of a top-level function
        size_t keep = -1;
        if (top_.find(lambda) != top_.end()) {
            for (size_t i = 0; i != size; ++i) {
                if (lambda->param(i)->type()->specialize(generic_map)->order() == 1) {
                    keep = i;
                    break;
                }
            }
        }

        size_t num = 0;
        for (size_t i = 0; i != size; ++i) {
            if (i != keep && lambda->param(i)->order() >= 1) {
                const Def* arg = ulambda->arg(i);
                indices[num] = i;
                with[num++] = arg;
                args[i] = arg;
            } else
                args[i] = 0;
        }
        with.shrink(num);
        indices.shrink(num);

        // check whether we can reuse an existing version
        auto args_i = args2lambda.find(args);
        Lambda* target;
        if (args_i != args2lambda.end()) 
            target = args_i->second; // use already dropped version as jump target 
        else
            args2lambda[args] = target = drop(scope, indices, with, generic_map);

        ulambda->jump(target, ulambda->args().cut(indices));
    }
}

size_t CFFLowering::process() {
    std::vector<Lambda*> todo;
    for (auto top : top_) {
        Scope scope(top);
        for (size_t i = scope.size(); i-- != 0;) {
            Lambda* lambda = scope[i];
            if (lambda->num_params()                                // is there sth to drop?
                && (lambda->is_generic()                            // drop generic stuff
                    || (!lambda->is_basicblock()                    // don't drop basic blocks
                        && (!lambda->is_returning()                 // drop non-returning lambdas
                            || top_.find(lambda) == top_.end()))))  // lift/drop returning non top-level lambdas
                todo.push_back(lambda);
        }
    }

    for (auto lambda : todo)
        transform(lambda);

    return todo.size();
}

void lower2cff(World& world) {
    size_t todo;
    do {
        CFFLowering lowering(world);
        todo = lowering.process();
        debug_verify(world);
        merge_lambdas(world);
        world.cleanup();
    } while (todo);
}

} // namespace anydsl2
