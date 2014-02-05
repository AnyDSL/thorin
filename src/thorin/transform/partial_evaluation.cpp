#include <iostream>
#include <unordered_map>
#include <queue>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/looptree.h"
#include "thorin/be/thorin.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/merge_lambdas.h"

namespace thorin {

class PartialEvaluator;

class LoopInfo {
public:
    LoopInfo() {}
    LoopInfo(PartialEvaluator* evaluator, const LoopHeader* loop)
        : evaluator_(evaluator)
        , loop_(loop)
        , evil_(false)
    {}

    const LoopHeader* loop() const { return loop_; }
    bool is_evil() { return evil_; }
    void set_evil() { evil_ = true; }

private:
    PartialEvaluator* evaluator_;
    const LoopHeader* loop_;
    bool evil_;
};

static std::vector<Lambda*> top_level_lambdas(World& world) {
    std::vector<Lambda*> result;
    auto scopes = top_level_scopes(world);
    for (auto scope : scopes)
        result.push_back(scope->entry());
    return result;
}

class EdgeType {
public:
    EdgeType(bool is_within, int n)
        : is_within_(is_within)
        , n_(n)
    {}

    bool is_within() const { return is_within_; }
    bool is_cross() const { return !is_within(); }
    int n() const { return n_; }

private:
    bool is_within_;
    int n_;
};

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : world_(world)
        , scope_(world, top_level_lambdas(world))
        , loops_(scope_)
    {
        loops_.dump();
        collect_headers(loops_.root());
        for (auto lambda : world.lambdas())
            new2old_[lambda] = lambda;
    }

    void collect_headers(const LoopNode*);
    void process();
    void rewrite_jump(Lambda* lambda, Lambda* to, ArrayRef<size_t> idxs);
    void remove_runs(Lambda* lambda);
    void update_new2old(const Def2Def& map);
    EdgeType classify(Lambda* src, Lambda* dst) const;
    int order(EdgeType) const;
    int order(Lambda* src, Lambda* dst) const { return order(classify(src, dst)); }

    const LoopHeader* is_header(Lambda* lambda) const {
        auto i = lambda2header_.find(new2old_[lambda]);
        if (i != lambda2header_.end())
            return i->second;
        return nullptr;
    }

    void push_loop(const LoopHeader* header) {
        loop_stack_.emplace_back(this, header);
    }
    void pop_loop(int ord) {
        //loop_stack_.resize(loop_stack_.size() - (ord - (EXIT-1)));
    }

    LoopInfo& top_loop() {
        return loop_stack_.back();
    }

    void push(Lambda* src, Lambda* dst, bool evil = false) {
        std::cout << "pushing: " << std::endl;
        std::cout << dst->unique_name() << std::endl;
        int ord = order(src, dst);
        auto& ord2stack_ = pos_ord2stack_;
        if (ord < 0) {
            ord2stack_ = neg_ord2stack_;
            ord = -ord;
        }

        if (size_t(ord) >= ord2stack_.size())
            ord2stack_.resize(ord+1);
        ord2stack_[ord].push_back(dst);

        //if (evil && ord >= EXIT) {
            //ord -= (EXIT-1);
            //for (auto i = loop_stack_.rbegin(), e = loop_stack_.rend(); i != e && ord >= 0; ++i, --ord) {
                //i->set_evil();
            //}
        //}
    }

    Lambda* pop() {
        std::vector<std::vector<Lambda*>>* a_ord2stack[2] = { &neg_ord2stack_, &pos_ord2stack_ };
        for (auto& ord2stack : a_ord2stack) {
            for (auto& stack : *ord2stack) {
                if (!stack.empty()) {
                    auto result = stack.back();
                    stack.pop_back();
                    return result;
                }
            }
        }
        return nullptr;
    }

    World& world_;
    Scope scope_;
    LoopTree loops_;
    Lambda2Lambda new2old_;
    std::unordered_map<Lambda*, const LoopHeader*> lambda2header_;
    std::unordered_set<Lambda*> done_;
    std::vector<LoopInfo> loop_stack_;
    std::vector<std::vector<Lambda*>> neg_ord2stack_;
    std::vector<std::vector<Lambda*>> pos_ord2stack_;
};

EdgeType PartialEvaluator::classify(Lambda* nsrc, Lambda* ndst) const {
    auto src = new2old_[nsrc];
    auto dst = new2old_[ndst];
    auto hsrc = loops_.lambda2header(src);
    auto hdst = loops_.lambda2header(dst);

    if (loops_.contains(hsrc, ndst))
        return EdgeType(true, hdst->depth() - hsrc->depth()); // within n, n positive
    if (loops_.contains(hdst, nsrc))
        return EdgeType(true, hsrc->depth() - hdst->depth()); // within n, n negative
    else {
#ifndef NDEBUG
        for (auto i = hsrc; i != hdst; i = i->parent())
            assert(!i->is_root());
#endif
        return EdgeType(false, hsrc->depth() - hdst->depth());// cross n, n <= 0
    }
}

int PartialEvaluator::order(EdgeType e) const {
    auto level = loop_stack_.size();
    if (e.is_within())
        return -2*(e.n() + level) + 1;
    return -2*(e.n() + level);
}

void PartialEvaluator::collect_headers(const LoopNode* n) {
    if (const LoopHeader* header = n->isa<LoopHeader>()) {
        for (auto lambda : header->lambdas())
            lambda2header_[lambda] = header;
        for (auto child : header->children())
            collect_headers(child);
    }
}

void PartialEvaluator::process() {
    for (auto src : top_level_lambdas(world_)) {
        do {
            if (done_.find(src) != done_.end())
                continue;
            done_.insert(src);

            std::cout << "src: " << src->unique_name() << std::endl;
            std::cout << "loop stack:" << std::endl;
            for (auto& info : loop_stack_) {
                std::cout << info.loop()->lambdas().front()->unique_name() << std::endl;
            }
            std::cout << "----" << std::endl;
            emit_thorin(world_);
            assert(!src->empty());

            auto succs = src->direct_succs();
            bool fold = false;

            auto to = src->to();
            if (auto run = to->isa<Run>()) {
                to = run->def();
                fold = true;
            }

            Lambda* dst = to->isa_lambda();

            if (dst == nullptr) {
                for (auto succ : succs)
                    push(src, succ, true);
                continue;
            } else {
                for (auto succ : succs)
                    if (succ != dst)
                        push(src, succ, true);
            }

            std::vector<Def> f_args, r_args;
            std::vector<size_t> f_idxs, r_idxs;

            for (size_t i = 0; i != src->num_args(); ++i) {
                if (auto evalop = src->arg(i)->isa<EvalOp>()) {
                    if (evalop->isa<Run>()) {
                        f_args.push_back(evalop);
                        r_args.push_back(evalop);
                        f_idxs.push_back(i);
                        r_idxs.push_back(i);
                        fold = true;
                    } else
                        assert(evalop->isa<Halt>());
                } else {
                    f_args.push_back(src->arg(i));
                    f_idxs.push_back(i);
                }
            }

            int ord = order(src, dst);
            if (ord == 42) {
                assert(!loop_stack_.empty());
                auto info = top_loop();
                if (info.is_evil())
                    continue;
            } else {
                //if (ord >= EXIT) {  // exting edge
                    //pop_loop(ord);
                //}

                if (auto header = is_header(dst)) {
                    std::cout << ord << std::endl;
                    std::cout << src->unique_name() << std::endl;
                    std::cout << dst->unique_name() << std::endl;
                    //assert(ord == NEXT_SCC);
                    push_loop(header);
                }

                if (!fold) {
                    push(src, dst);
                    continue;
                }
            }

            Scope scope(dst);
            Def2Def f_map;
            GenericMap generic_map;
            bool res = dst->type()->infer_with(generic_map, src->arg_pi());
            assert(res);
            auto f_to = drop(scope, f_map, f_idxs, f_args, generic_map);
            f_map[to] = f_to;
            update_new2old(f_map);

            if (f_to->to()->isa_lambda() 
                    || (f_to->to()->isa<Run>() && f_to->to()->as<Run>()->def()->isa_lambda())) {
                rewrite_jump(src, f_to, f_idxs);
                for (auto lambda : scope.rpo()) {
                    auto mapped = f_map[lambda]->as_lambda();
                    if (mapped != lambda)
                        mapped->update_to(world_.run(mapped->to()));
                }
                push(src, f_to);
            } else {
                Def2Def r_map;
                auto r_to = drop(scope, r_map, r_idxs, r_args, generic_map);
                r_map[to] = r_to;
                update_new2old(r_map);
                rewrite_jump(src, r_to, r_idxs);
                push(src, r_to);
            }
        } while ((src = pop()));
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
            assert(new2old_.contains(olambda));
            new2old_[nlambda] = new2old_[olambda];
        }
    }
}

//------------------------------------------------------------------------------

void partial_evaluation(World& world) { 
    emit_thorin(world);
    PartialEvaluator(world).process(); 
}

}
