#include "anydsl2/transform/cfg_builder.h"

#include <iostream>
#include <boost/unordered_set.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

struct Done {
    Array<const Def*> with;
    Lambda* lambda;

    Done(size_t size)
        : with(size)
        , lambda(0)
    {}

    Done(const Done& other)
        : with(other.with)
        , lambda(other.lambda)
    {}

    bool operator == (const Done& done) const { return with == done.with; }
};

size_t hash_value(const Done& done) { return hash_value(done.with); }

class CFGBuilder {
public:
    CFGBuilder(Lambda* entry)
        : scope(entry)
    {}

    void transform(LambdaSet& todo);
    Lambda* entry() const { return scope.entry(); }

private:

    Scope scope;

    typedef boost::unordered_set<Done> DoneSet;
    DoneSet done_entries;
};

void CFGBuilder::transform(LambdaSet& todo) {
    for_all (lambda, scope.rpo()) {
        if (lambda->is_ho() > 1) {
            size_t size = lambda->num_params();
            Done done(size);
            Array<size_t>  indices(size);

            // if there is only one use -> drop all parameters
            //bool full_mode = lambda->num_uses() == 1;
            bool full_mode = false;

            for_all (use, lambda->copy_uses()) {
                if (use.index() != 0 || !use.def()->isa<Lambda>())
                    continue;

                Lambda* ulambda = use.def()->as_lambda();
                // we are not allowed to modify our own recursive calls
                bool is_nested = scope.lambdas().find(ulambda) != scope.lambdas().end();
                if (ulambda->to() == entry() && is_nested)
                    continue;

                GenericMap generic_map;
                bool res = lambda->type()->infer_with(generic_map, ulambda->arg_pi());
                assert(res);
                
                size_t num = 0;
                for (size_t i = 0; i != size; ++i) {
                    if (full_mode || lambda->param(i)->type()->is_ho()) {
                        const Def* arg = ulambda->arg(i);
                        indices[num] = i;
                        done.with[num++] = arg;
                        // verify argument: do we have to perform an additional drop operation?
                        if (Lambda* argLambda = arg->isa_lambda()) {
                            if (argLambda->is_ho()) {
                                // we need to drop this lambda as well
                                todo.insert(argLambda);
                            }
                        }
                    }
                }

                // check whether we can reuse an existing version
                DoneSet::iterator de = done_entries.find(done);
                Lambda* target;
                if (de != done_entries.end()) {
                    // use already dropped version as jump target
                    target = de->lambda;
                } else {
                    target = lambda->drop(indices.slice_front(num), done.with.slice_front(num), generic_map, true);
                    scope.reassign_sids();
                    // store dropped entry with the specified arguments
                    done.lambda = target;
                    done_entries.insert(done);
                }
                ulambda->jump(target, ulambda->args().cut(indices.slice_front(num)));

                // remove from the todo list
                todo.erase(lambda);
            }
        }
    }
}

void cfg_transform(World& world) {
    LambdaSet todo = find_root_lambdas(world.lambdas());
    while (todo.size() > 0) {
        // we need to drop an additional lambda
        Lambda* lambda = *todo.begin();
        // remove from todo list
        todo.erase(lambda);
        // transform required lambda
        CFGBuilder builder(lambda);
        builder.transform(todo);
    }
}

} // namespace anydsl2
