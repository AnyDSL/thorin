#include <iostream>
#include <unordered_map>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/type.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/merge_lambdas.h"

namespace thorin {

class CFFLowering {
public:
    CFFLowering(World& world) {
        for (auto scope : top_level_scopes(world))
            top_.insert(scope->entry());
    }

    void transform(Lambda* lambda);
    size_t process();

private:
    LambdaSet top_;
};

void CFFLowering::transform(Lambda* lambda) {
    Scope scope(lambda);
    std::unordered_map<Array<const DefNode*>, Lambda*> args2lambda;

    for (auto use : lambda->uses()) {
        if (use.index() != 0 || !use->isa<Lambda>())
            continue;

        Lambda* ulambda = use->as_lambda();
        if (scope.contains(ulambda))
            continue;

        GenericMap map;
        bool res = lambda->type()->infer_with(map, ulambda->arg_pi());
        assert(res);
        
        size_t size = lambda->num_params();
        Array<size_t> indices(size);
        Array<Def> with(size);
        Array<const DefNode*> args(size);

        // don't drop the "return" of a top-level function
        size_t keep = -1;
        if (top_.find(lambda) != top_.end()) {
            for (size_t i = 0; i != size; ++i) {
                if (lambda->param(i)->type()->specialize(map)->order() == 1) {
                    keep = i;
                    break;
                }
            }
        }

        size_t num = 0;
        for (size_t i = 0; i != size; ++i) {
            if (i != keep && lambda->param(i)->order() >= 1) {
                Def arg = ulambda->arg(i);
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
            args2lambda[args] = target = drop(scope, indices, with, map);

        ulambda->jump(target, ulambda->args().cut(indices));
    }
}

size_t CFFLowering::process() {
    std::vector<Lambda*> todo;
    for (auto top : top_) {
        Scope scope(top);
        for (auto i = scope.rbegin(), e = scope.rend(); i != e; ++i) {
            auto lambda = *i;
            // check for builtin functionality
            if (lambda->is_builtin() || lambda->is_connected_to_builtin())
                continue;
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

} // namespace thorin
