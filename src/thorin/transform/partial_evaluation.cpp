#include <iostream>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/looptree.h"
#include "thorin/be/thorin.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/merge_lambdas.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : world(world)
        , top(top_lambda(world))
        , scope(top)
        , loops(scope)
    {
        collect_headers(loops.root());
    }

    void collect_headers(const LoopNode*);
    void eval(Lambda*);
    void rewrite_jump(Lambda* lambda, Lambda* to, ArrayRef<size_t> idxs);

    World& world;
    Lambda* top;
    Scope scope;
    LoopTree loops;
    std::unordered_set<Lambda*> headers;
};

void PartialEvaluator::collect_headers(const LoopNode* n) {
    if (const LoopHeader* header = n->isa<LoopHeader>()) {
        for (auto lambda : header->lambdas())
            headers.insert(lambda);
        for (auto child : header->children())
            collect_headers(child);
    }
}

void PartialEvaluator::eval(Lambda* lambda) {
    if (lambda->to()->isa<Halt>())
        return;

    Lambda* to;
    if (auto run = lambda->to()->isa<Run>())
        to = run->def()->isa_lambda();
    else
        to = lambda->to()->isa_lambda();

    if (to == nullptr)
        return;

    std::vector<Def> f_args, r_args;
    std::vector<size_t> f_idxs, r_idxs;

    for (size_t i = 0; i != lambda->num_args(); ++i) {
        if (auto evalop = lambda->arg(i)->isa<EvalOp>()) {
            if (evalop->isa<Run>()) {
                f_args.push_back(evalop);
                r_args.push_back(evalop);
                f_idxs.push_back(i);
                r_idxs.push_back(i);
            } else
                assert(evalop->isa<Halt>());
        } else {
            f_args.push_back(lambda->arg(i));
            f_idxs.push_back(i);
        }
    }

    Scope scope(to);
    auto f_to = drop(scope, f_idxs, f_args);
    auto r_to = drop(scope, r_idxs, r_args);

    if (f_to->to()->isa_lambda())
        goto full;

    if (auto run = f_to->to()->isa<Run>())
        if (run->def()->isa_lambda())
            goto full;

run_only:
    rewrite_jump(lambda, r_to, r_idxs);
    return;

full:
    rewrite_jump(lambda, f_to, f_idxs);
    eval(f_to);
}

void PartialEvaluator::rewrite_jump(Lambda* lambda, Lambda* to, ArrayRef<size_t> idxs) {
    std::vector<Def> new_args;
    size_t x = 0;
    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
        if (x < idxs.size() && i == idxs[x])
            ++x;
        else
            new_args.push_back(lambda->arg(i));
    }

    lambda->jump(to, new_args);
}
//------------------------------------------------------------------------------

void partial_evaluation(World& world) { 
    PartialEvaluator pe(world);
    for (auto lambda : world.lambdas()) {
        if (lambda->to()->isa<Run>()) {
            std::cout << lambda->unique_name() << std::endl;
            pe.eval(lambda);
        }
    }
    return; 
}

}
