#include <iostream>
#include <queue>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/looptree.h"
#include "thorin/be/thorin.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/merge_lambdas.h"

namespace thorin {

class Branch {
public:
    Branch(const std::vector<Lambda*>& succs)
        : index_(0)
        , succs_(succs)
    {
        assert(!succs_.empty());
    }
    Branch(std::initializer_list<Lambda*> succs)
        : index_(0)
        , succs_(succs)
    {
        assert(!succs_.empty());
    }

    Lambda* cur() const { return succs_[index_]; }
    bool inc() { 
        assert(index_ < succs_.size()); 
        ++index_; 
        return index_ == succs_.size();
    }

private:
    size_t index_;
    std::vector<Lambda*> succs_;
};

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : world(world)
        , scope(world)
        , loops(scope)
    {
        //loops.dump();
        collect_headers(loops.root());
        for (auto lambda : world.lambdas())
            new2old[lambda] = lambda;
    }

    void collect_headers(const LoopNode*);
    void process();
    void rewrite_jump(Lambda* lambda, Lambda* to, ArrayRef<size_t> idxs);
    void remove_runs(Lambda* lambda);
    void update_new2old(const Def2Def& map);

    World& world;
    Scope scope;
    LoopTree loops;
    Lambda2Lambda new2old;
    std::unordered_set<Lambda*> headers;
    std::unordered_set<Lambda*> done;
    std::vector<Branch> branches;
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
    for (auto top : top_level_lambdas(world)) {
        branches.push_back(Branch({top}));

        while (!branches.empty()) {
            auto& branch = branches.back();
            auto cur = branch.cur();
            if (branch.inc())
                branches.pop_back();

            if (done.find(cur) != done.end())
                continue;
            done.insert(cur);

            std::cout << "cur: " << cur->unique_name() << std::endl;
            emit_thorin(world);
            assert(!cur->empty());

            auto succs = cur->direct_succs();
            bool fold = false;

            auto to = cur->to();
             if (auto run = to->isa<Run>()) {
                to = run->def();
                fold = true;
             }

            Lambda* dst = to->isa_lambda();

            if (dst == nullptr) {
                if (!succs.empty())
                    branches.emplace_back(succs);
                continue;
            }

            std::vector<Def> f_args, r_args;
            std::vector<size_t> f_idxs, r_idxs;

            for (size_t i = 0; i != cur->num_args(); ++i) {
                if (auto evalop = cur->arg(i)->isa<EvalOp>()) {
                    if (evalop->isa<Run>()) {
                        f_args.push_back(evalop);
                        r_args.push_back(evalop);
                        f_idxs.push_back(i);
                        r_idxs.push_back(i);
                        fold = true;
                    } else
                        assert(evalop->isa<Halt>());
                } else {
                    f_args.push_back(cur->arg(i));
                    f_idxs.push_back(i);
                }
            }

            if (!fold) {
                branches.push_back(Branch({dst}));
                continue;
            }

            Scope scope(dst);
            Def2Def f_map;
            auto f_to = drop(scope, f_map, f_idxs, f_args);
            f_map[to] = f_to;
            update_new2old(f_map);

            if (f_to->to()->isa_lambda() 
                    || (f_to->to()->isa<Run>() && f_to->to()->as<Run>()->def()->isa_lambda())) {
                rewrite_jump(cur, f_to, f_idxs);
                for (auto lambda : scope.rpo()) {
                    auto mapped = f_map[lambda]->as_lambda();
                    if (mapped != lambda)
                        mapped->update_to(world.run(mapped->to()));
                }
                branches.push_back(Branch({f_to}));
            } else {
                Def2Def r_map;
                auto r_to = drop(scope, r_map, r_idxs, r_args);
                r_map[to] = r_to;
                update_new2old(r_map);
                rewrite_jump(cur, r_to, r_idxs);
                branches.push_back(Branch({r_to}));
            }
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
            //std::cout << nlambda->unique_name() << " -> "  << olambda->unique_name() << std::endl;
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
