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

enum {
    NEXT_SCC = 0,
    FORWARD = 1,
    BACK = 2,
    EXIT = 3
};

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
    int order(Lambda* src, Lambda* dst) const;

    const LoopHeader* is_header(Lambda* lambda) const {
        auto i = lambda2header_.find(new2old_[lambda]);
        if (i != lambda2header_.end())
            return i->second;
        return nullptr;
    }

    const LoopHeader* on_stack(Lambda* lambda) const {
        auto parent = loops_.lambda2header(new2old_[lambda]);
        for (auto& info : loop_stack_) {
            if (info.loop() == parent)
                return parent;
        }
        return nullptr;
    }

    void push_loop(const LoopHeader* header) {
        loop_stack_.emplace_back(this, header);
    }
    void pop_loop(int ord) {
        loop_stack_.resize(loop_stack_.size() - (ord - (EXIT-1)));
    }
    LoopInfo& top_loop() {
        return loop_stack_.back();
    }

    Lambda* pop() {
        for (auto& stack : ord2stack_) {
            if (!stack.empty()) {
                auto result = stack.back();
                stack.pop_back();
                return result;
            }
        }
        return nullptr;
    }

    void push(Lambda* src, Lambda* dst, bool evil = false) {
        int ord = order(src, dst);
        if (size_t(ord) >= ord2stack_.size())
            ord2stack_.resize(ord+1);
        ord2stack_[ord].push_back(dst);

        if (evil && ord >= EXIT) {
            ord -= (EXIT-1);
            for (auto i = loop_stack_.rbegin(), e = loop_stack_.rend(); i != e && ord >= 0; ++i, --ord) {
                i->set_evil();
            }
        }
    }

    World& world_;
    Scope scope_;
    LoopTree loops_;
    Lambda2Lambda new2old_;
    std::unordered_map<Lambda*, const LoopHeader*> lambda2header_;
    std::unordered_set<Lambda*> done_;
    std::vector<LoopInfo> loop_stack_;
    std::vector<std::vector<Lambda*>> ord2stack_;
};

int PartialEvaluator::order(Lambda* nsrc, Lambda* ndst) const {
    auto src = new2old_[nsrc];
    auto dst = new2old_[ndst];
    auto hsrc = loops_.lambda2header(src);
    auto hdst = loops_.lambda2header(dst);

    if (hsrc == hdst) {     // same SCC?
        if (hsrc->headers().contains(dst))
            return BACK;    // backedge
        return FORWARD;     // forward jump
    }

    int result = hsrc->depth() - hdst->depth();

    if (hdst->parent() == hsrc) {
        assert(result == -1);
        return NEXT_SCC;    // jump to next SCC nesting level
    }

    assert(result > 0);
    return result+2;        // disambiguate from backedge
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
    for (auto cur : top_level_lambdas(world_)) {
        do {
            if (done_.find(cur) != done_.end())
                continue;
            done_.insert(cur);

            std::cout << "cur: " << cur->unique_name() << std::endl;
            emit_thorin(world_);
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
                for (auto succ : succs)
                    push(cur, succ, true);
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

            int ord = order(cur, dst);
            if (ord == BACK) {
                auto info = top_loop();
                if (info.is_evil())
                    continue;
            } else {
                if (ord >= EXIT) {  // exting edge
                    pop_loop(ord);
                }

                if (auto header = is_header(dst)) {
                    std::cout << ord << std::endl;
                    std::cout << cur->unique_name() << std::endl;
                    std::cout << dst->unique_name() << std::endl;
                    assert(ord == NEXT_SCC);
                    push_loop(header);
                }

                if (!fold) {
                    push(cur, dst);
                    continue;
                }
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
                        mapped->update_to(world_.run(mapped->to()));
                }
                push(cur, f_to);
            } else {
                Def2Def r_map;
                auto r_to = drop(scope, r_map, r_idxs, r_args);
                r_map[to] = r_to;
                update_new2old(r_map);
                rewrite_jump(cur, r_to, r_idxs);
                push(cur, r_to);
            }
        } while ((cur = pop()));
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
