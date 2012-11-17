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
    CFGBuilder(World& world)
        : world(world)
        , top(find_root_lambdas(world.lambdas()))
    {}

    void transform(Lambda* lambda);
    void process();

private:

    World& world;
    LambdaSet top;
};

void CFGBuilder::transform(Lambda* lambda) {
    typedef boost::unordered_set<Done> DoneSet;
    DoneSet done_entries;
    Scope scope(lambda);
    size_t size = lambda->num_params();
    Done done(size);
    Array<size_t>  indices(size);

    // if there is only one use -> drop all parameters
    bool full_mode = lambda->num_uses() == 1;

    for_all (use, lambda->copy_uses()) {
        if (use.index() != 0 || !use.def()->isa<Lambda>())
            continue;

        Lambda* ulambda = use.def()->as_lambda();
        GenericMap generic_map;
        bool res = lambda->type()->infer_with(generic_map, ulambda->arg_pi());
        assert(res);
        
        size_t num = 0;
        for (size_t i = 0; i != size; ++i) {
            if (full_mode || lambda->param(i)->order() >= 1) {
                const Def* arg = ulambda->arg(i);
                indices[num] = i;
                done.with[num++] = arg;
            }
        }

        // check whether we can reuse an existing version
        DoneSet::iterator de = done_entries.find(done);
        Lambda* target;
        if (de != done_entries.end()) {
            // use already dropped version as jump target
            target = de->lambda;
        } else {
//#if 0
            if (lambda->num_uses() > 1 && lambda->is_returning() && top.find(lambda) == top.end()) {
                FreeVariables fv = scope.free_variables();
                for_all (def, fv) {
                    if (def->order() > 0)
                        goto do_dropping;
                }

                target = scope.lift(fv);
                ulambda->jump(target, Array<const Def*>(ulambda->args(), fv));
                return;
            }

do_dropping:
//#endif
            target = scope.drop(indices.slice_front(num), done.with.slice_front(num), true, generic_map);
            // store dropped entry with the specified arguments
            done.lambda = target;
            done_entries.insert(done);
        }
        ulambda->jump(target, ulambda->args().cut(indices.slice_front(num)));
    }
}

void CFGBuilder::process() {
    std::vector<Lambda*> todo;
    for_all (top_lambda, top) {
        Scope scope(top_lambda);
        for (size_t i = scope.size()-1; i != size_t(-1); --i) {
            Lambda* lambda = scope.rpo(i);
            if (lambda->num_params()                                // is there sth to drop?
                    && (lambda->num_uses() == 1                     // just 1 user -- always drop
                        || (!lambda->is_bb()                        // don't drop basic blocks
                            && (!lambda->is_returning()             // drop non-returning lambdas
                                || top.find(lambda) == top.end()))))// lift/drop returning non top-level lambdas
                todo.push_back(lambda);
        }
    }

    for_all (lambda, todo)
        transform(lambda);
}

void cfg_transform(World& world) {
    CFGBuilder builder(world);
    builder.process();
}

} // namespace anydsl2
