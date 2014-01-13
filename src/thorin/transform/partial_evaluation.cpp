#include <iostream>
#include <queue>

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
        , loops(world)
    {
        loops.dump();
        collect_headers(loops.root());
        for (auto lambda : world.lambdas())
            new2old[lambda] = lambda;
    }

    void collect_headers(const LoopNode*);
    void process();
    void eval(Lambda*);
    void rewrite_jump(Lambda* lambda, Lambda* to, ArrayRef<size_t> idxs);
    void remove_runs(Lambda* lambda);
    void update_new2old(const Def2Def& map);

    World& world;
    LoopTree loops;
    Lambda2Lambda new2old;
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

void PartialEvaluator::process() {
    while (true) {
        Scope scope(world);
        for (auto lambda : scope.rpo()) {
            for (auto op : lambda->ops()) {
                if (op->isa<Run>()) {
                    eval(lambda);
                    world.cleanup();
                    std::cout << "-------------------- cleanup ----------------" << std::endl;
                    std::cout << "--- new2old ---" << std::endl;
                    for (auto p : new2old) {
                        auto nlambda = p.first ->as_lambda();
                        auto olambda = p.second->as_lambda();
                        std::cout << nlambda->unique_name() << " -> "  << olambda->unique_name() << std::endl;
                    }
                    std::cout << "--- new2old ---" << std::endl;
                    emit_thorin(world);
                    goto outer_loop;
                }
            }
        }

        break;
outer_loop:;
    }
}

void PartialEvaluator::eval(Lambda* lambda) {
    while (true) {
        std::cout << "--------------------------" << std::endl;
        std::cout << lambda->unique_name() << std::endl;
        std::cout << "--------------------------" << std::endl;
        emit_thorin(world);
        assert(!lambda->empty());

        if (lambda->to()->isa<Halt>()) {
            remove_runs(lambda);
            return;
        }

        Lambda* to;
        if (auto run = lambda->to()->isa<Run>())
            to = run->def()->isa_lambda();
        else
            to = lambda->to()->isa_lambda();

        if (to == nullptr) {
            remove_runs(lambda);
            return;
        }

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
        Def2Def f_map, r_map;
        auto f_to = drop(scope, f_map, f_idxs, f_args);
        auto r_to = drop(scope, r_map, r_idxs, r_args);
        f_map[to] = f_to;
        r_map[to] = r_to;
        update_new2old(f_map);
        update_new2old(r_map);

        if (f_to->to()->isa_lambda() 
                || (f_to->to()->isa<Run>() && f_to->to()->as<Run>()->def()->isa_lambda())) {
            rewrite_jump(lambda, f_to, f_idxs);
            for (auto lambda : scope.rpo()) {
                auto mapped = f_map[lambda]->as_lambda();
                if (mapped != lambda)
                    mapped->update_to(world.run(mapped->to()));
            }
            lambda = f_to;
        } else {
            rewrite_jump(lambda, r_to, r_idxs);
            return;
        }
    }
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

void PartialEvaluator::remove_runs(Lambda* lambda) {
    for (size_t i = 0, e = lambda->size(); i != e; ++i) {
        if (auto run = lambda->op(i)->isa<Run>())
            lambda->update_op(i, run->def());
    }
}

void PartialEvaluator::update_new2old(const Def2Def& old2new) {
    for (auto p : old2new) {
        if (auto olambda = p.first->isa_lambda()) {
            auto nlambda = p.second->as_lambda();
            std::cout << nlambda->unique_name() << " -> "  << olambda->unique_name() << std::endl;
            assert(new2old.contains(olambda));
            new2old[nlambda] = new2old[olambda];
        }
    }
}

//------------------------------------------------------------------------------

void partial_evaluation(World& world) { 
    emit_thorin(world);
    PartialEvaluator(world).process(); 
}

}
