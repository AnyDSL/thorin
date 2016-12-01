#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/log.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(Scope& top_scope)
        : top_scope_(top_scope)
    {}
    ~PartialEvaluator() {
        top_scope(); // trigger update if dirty
    }

    World& world() { return top_scope_.world(); }
    void run();
    void eval(Continuation* begin, Continuation* end);
    Continuation* postdom(Continuation*, const Scope&);
    Continuation* postdom(Continuation*);
    void enqueue(Continuation* continuation) {
        if (top_scope().contains(continuation)) {
            auto p = visited_.insert(continuation);
            if (p.second)
                queue_.push(continuation);
        }
    }

    void init_cur_scope(Continuation* entry) {
        cur_scope_ = new Scope(entry);
        cur_dirty_ = false;
    }

    void release_cur_scope() {
        delete cur_scope_;
    }

    const Scope& cur_scope() {
        if (cur_dirty_) {
            cur_dirty_ = false;
            return cur_scope_->update();
        }
        return *cur_scope_;
    }

    const Scope& top_scope() {
        if (top_dirty_) {
            top_dirty_ = false;
            return top_scope_.update();
        }
        return top_scope_;
    }

    void mark_dirty() { top_dirty_ = cur_dirty_ = true; }
    Continuation* get_continuation(Continuation* continuation) { return continuation->callee()->as<EvalOp>()->end()->isa_continuation(); }

private:
    Scope* cur_scope_;
    Scope& top_scope_;
    ContinuationSet done_;
    std::queue<Continuation*> queue_;
    ContinuationSet visited_;
    HashMap<Call, Continuation*> cache_;
    bool cur_dirty_;
    bool top_dirty_ = false;
};

void PartialEvaluator::run() {
    enqueue(top_scope().entry());

    while (!queue_.empty()) {
        auto continuation = pop(queue_);

        // due to the optimization below to eat up a call, we might see a new Run here
        while (continuation->callee()->isa<Run>()) {
            auto cur = continuation->callee();
            init_cur_scope(continuation);
            eval(continuation, get_continuation(continuation));
            release_cur_scope();
            if (cur == continuation->callee())
                break;
        }

        for (auto succ : top_scope().f_cfg().succs(continuation))
            enqueue(succ->continuation());
    }
}

void PartialEvaluator::eval(Continuation* cur, Continuation* end) {
    if (end == nullptr)
        WLOG("no matching end: % at %", cur, cur->loc());
    else
        DLOG("eval: % -> %", cur, end);

    while (true) {
        if (cur == nullptr) {
            WLOG("cur is nullptr");
            return;
        } else if (cur->empty()) {
            WLOG("empty: %", cur);
            return;
        } else if (done_.contains(cur)) {
            DLOG("already done: %", cur);
            return;
        } else
            DLOG("cur: %", cur);

        done_.insert(cur);

        Continuation* dst = nullptr;
        if (auto run = cur->callee()->isa<Run>()) {
            dst = run->begin()->isa_continuation();
        } else if (cur->callee()->isa<Hlt>()) {
            cur = get_continuation(cur);
            continue;
        } else {
            dst = cur->callee()->isa_continuation();
        }

        if (dst == nullptr || dst->empty()) {
            cur = postdom(cur);
            if (cur == nullptr)
                return;
            if (end == nullptr)
                continue;

            const auto& postdomtree = top_scope().b_cfg().domtree();
            auto ncur = top_scope().cfa(cur);
            auto nend = top_scope().cfa(end);

            assert(ncur != nullptr);
            if (nend == nullptr) {
                WLOG("end became unreachable: %", end);
                continue;
            }

            for (auto i = nend; i != postdomtree.root(); i = postdomtree.idom(i)) {
                if (i == ncur) {
                    DLOG("overjumped end: %", cur);
                    return;
                }
            }

            if (cur == end) {
                DLOG("end: %", end);
                return;
            }
            continue;
        }

        Array<const Def*> ops(cur->num_ops());
        ops.front() = dst;
        bool all = true;
        for (size_t i = 1, e = ops.size(); i != e; ++i) {
            if (!cur->op(i)->isa<Hlt>())
                ops[i] = cur->op(i);
            else
                all = false;
        }

        Call call(ops);

        bool go_out = dst == end;
        DLOG("dst: %", dst);

        if (auto cached = find(cache_, call)) {             // check for cached version
            jump_to_cached_call(cur, cached, call);
            DLOG("using cached call: %", cur);
            return;
        } else {                                            // no cached version found... create a new one
            Scope scope(call.callee()->as_continuation());
            Mangler mangler(scope, call.args(), Defs());
            auto dropped = mangler.mangle();
            if (end != nullptr) {
                if (auto nend = mangler.def2def(end)) {
                    if (end != nend) {
                        DLOG("changed end: % -> %", end, nend);
                        end = nend->as_continuation();
                    }
                }
            }

            if (dropped->callee() == world().branch()) {
                // TODO don't stupidly inline functions
                // TODO also don't peel inside functions with incoming back-edges
            }

            mark_dirty();
            cache_[call] = dropped;
            jump_to_cached_call(cur, dropped, call);
            if (all) {
                cur->jump(dropped->callee(), dropped->args(), cur->jump_loc());
                done_.erase(cur);
            } else
                cur = dropped;
        }

        if (dst == end || go_out) {
            DLOG("end: %", end);
            return;
        }
    }
}

Continuation* PartialEvaluator::postdom(Continuation* cur) {
    auto is_valid = [&] (Continuation* continuation) {
        auto p = (continuation && !continuation->empty()) ? continuation : nullptr;
        if (p)
            DLOG("postdom: % -> %", cur, p);
        return p;
    };

    if (top_scope_.entry() != cur_scope_->entry()) {
        if (auto p = is_valid(postdom(cur, cur_scope())))
            return p;
    }

    if (auto p = is_valid(postdom(cur, top_scope())))
        return p;

    WLOG("no postdom found for % at %", cur, cur->loc());
    return nullptr;
}

Continuation* PartialEvaluator::postdom(Continuation* cur, const Scope& scope) {
    const auto& postdomtree = scope.b_cfg().domtree();
    if (auto n = scope.cfa(cur))
        return postdomtree.idom(n)->continuation();
    return nullptr;
}

//------------------------------------------------------------------------------

void eval(World& world) {
    Scope::for_each(world, [&] (Scope& scope) { PartialEvaluator(scope).run(); });
}

void partial_evaluation(World& world) {
    world.cleanup();
    ILOG_SCOPE(eval(world));

    for (auto primop : world.primops()) {
        if (auto evalop = primop->isa<EvalOp>())
            evalop->replace(evalop->begin());
    }
}

//------------------------------------------------------------------------------

}
