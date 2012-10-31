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
        : with(size), lambda(0)
    { }

    Done(const Done& other)
        : with(other.with), lambda(other.lambda)
    { }

    bool operator == (const Done& done) const { return with == done.with; }
};

size_t hash_value(const Done& done) { return hash_value(done.with); }

class CFGBuilder {
public:
    CFGBuilder(Lambda* entry)
        : entry(entry), scope(entry)
    { }

    void transform(LambdaSet& to_handle);

private:
    typedef boost::unordered_set<Done> DoneSet;

    Lambda* entry;
    Scope scope;
    DoneSet done_entries;
};

void CFGBuilder::transform(LambdaSet& to_handle) {
    for_all (lambda, scope.rpo()) {
        if (lambda->is_ho()) {
            size_t size = lambda->num_params();
            Done done(size);
            Array<size_t>  indices(size);

            // if there is only one use -> drop all parameters
            bool full_mode = lambda->num_uses() == 1;

            for_all (use, lambda->uses()) {
                if (use.index() != 0)
                    continue;

                if (Lambda* ulambda = use.def()->isa_lambda()) {
                    size_t num = 0;
                    // we are not allowed to modify our own recursive calls
                    bool is_nested = scope.lambdas().find(ulambda) != scope.lambdas().end();
                    if(ulambda->to() == entry && is_nested)
                        continue;
                    
                    for (size_t i = 0; i != size; ++i) {
                        if (full_mode || lambda->param(i)->type()->is_ho()) {
                            const Def* arg = ulambda->arg(i);
                            indices[num] = i;
                            done.with[num++] = arg;
                            // verify argument: do we have to perform an additional drop operation?
                            if(Lambda* argLambda = arg->isa_lambda()) {
                                if(argLambda->is_ho()) {
                                    // we need to drop this lambda as well
                                    to_handle.insert(argLambda);
                                }
                            }
                        }
                    }
                    // check whether we can reuse an existing version
                    DoneSet::iterator de = done_entries.find(done);
                    Lambda* target;
                    if(de != done_entries.end()) {
                        // use already dropped version as jump target
                        target = de->lambda;
                    } else {
                        target = lambda->drop(indices.slice_front(num), done.with.slice_front(num), true);
                        scope.reassign_sids();
                        // store dropped entry with the specified arguments
                        done.lambda = target;
                        done_entries.insert(done);
                    }
                    ulambda->jump(target, ulambda->args().cut(indices.slice_front(num)));

                    // remove from the to_handle list
                    to_handle.erase(lambda);
                }
            }
        }
    }
}

void cfg_transform(World& world) {
    LambdaSet to_transform = find_root_lambdas(world.lambdas());
    while(to_transform.size() > 0) {
        // we need to drop an additional lambda
        Lambda* lambda = *to_transform.begin();
        // remove from todo list
        to_transform.erase(lambda);
        // transform required lambda
        CFGBuilder builder(lambda);
        builder.transform(to_transform);
    }
}

} // namespace anydsl2
