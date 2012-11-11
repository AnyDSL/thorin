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
    CFGBuilder(Lambda* lambda)
        : lambda(lambda)
    {}

    void transform();

private:

    Lambda* lambda;

    typedef boost::unordered_set<Done> DoneSet;
    DoneSet done_entries;
};

void CFGBuilder::transform() {
    size_t size = lambda->num_params();
    Done done(size);
    Array<size_t>  indices(size);

    // if there is only one use -> drop all parameters
    //bool full_mode = lambda->num_uses() == 1;
    bool full_mode = false;

    for_all (use, lambda->copy_uses()) {
        if (use.index() != 0 || !use.def()->isa<Lambda>())
            continue;

        std::cout << "def: " << lambda->debug << std::endl;
        std::cout << "\t"; 
        use.def()->dump();

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
            target = lambda->drop(indices.slice_front(num), done.with.slice_front(num), generic_map, true);
            // store dropped entry with the specified arguments
            done.lambda = target;
            done_entries.insert(done);
        }
        ulambda->jump(target, ulambda->args().cut(indices.slice_front(num)));
    }
}

void cfg_transform(World& world) {
    LambdaSet top = find_root_lambdas(world.lambdas());
    LambdaSet todo;
    for_all (lambda, world.lambdas()) {
        //if (lambda->num_uses() == 1) {
            //todo.insert(lambda);
            //continue;
        //}
        if (lambda->is_bb())
            continue;
        if (lambda->is_returning()) {
            if (top.find(lambda) == todo.end())
                todo.insert(lambda);
            continue;
        } else {
            todo.insert(lambda);
        }
    }

    std::cout << "todos: " << std::endl;
    for_all (l, todo)
        std::cout << l->debug << std::endl;
    std::cout << "---" << std::endl;

    for_all (lambda, todo) {
        CFGBuilder builder(lambda);
        builder.transform();
        for_all (lambda, world.lambdas()) {
            //if (lambda->num_uses() == 1) {
                //todo.insert(lambda);
                //continue;
            //}
            if (lambda->is_bb())
                continue;
            if (lambda->is_returning()) {
                if (top.find(lambda) == todo.end())
                    todo.insert(lambda);
                continue;
            } else {
                todo.insert(lambda);
            }
        }
    }
}

} // namespace anydsl2
