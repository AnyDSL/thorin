#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/log.h"
#include "thorin/util/queue.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(Scope& scope)
        : scope_(scope)
    {}

    const Scope& scope() {
        if (dirty_) {
            dirty_ = false;
            return scope_.update();
        }
        return scope_;
    }

    World& world() { return scope().world(); }
    void run();
    void eval(Lambda* begin, Lambda* end);
    Lambda* postdom(Lambda*);
    void enqueue(Lambda* lambda) {
        if (scope().outer_contains(lambda)) {
            auto p = visited_.insert(lambda);
            if (p.second)
                queue_.push(lambda);
        }
    }

private:
    Scope& scope_;
    LambdaSet done_;
    std::queue<Lambda*> queue_;
    LambdaSet visited_;
    HashMap<Array<Def>, Lambda*> cache_;
    bool dirty_ = false;
};

static Lambda* continuation(Lambda* lambda) {
    return lambda->num_args() != 0 ? lambda->args().back()->isa_lambda() : (Lambda*) nullptr;
}

void PartialEvaluator::run() {
    enqueue(scope().entry());

    while (!queue_.empty()) {
        auto lambda = pop(queue_);

        if (lambda->to()->isa<Run>())
            eval(lambda, continuation(lambda));

        for (auto succ : scope().f_cfg().succs(lambda))
            enqueue(succ->lambda());
    }
}

void PartialEvaluator::eval(Lambda* cur, Lambda* end) {
    if (end == nullptr)
        WLOG("no matching end: % at %", cur, cur->loc());
    else
        DLOG("eval: % -> %", cur, end);

    while (true) {
        if (cur == end) {
            DLOG("end: %", end);
            return;
        } else if (done_.contains(cur)) {
            DLOG("already done: %", cur);
            return;
        } else if (cur == nullptr) {
            WLOG("cur is nullptr");
            return;
        } else if (cur->empty()) {
            WLOG("empty: %", cur);
            return;
        }

        done_.insert(cur);

        Lambda* dst = nullptr;
        if (auto run = cur->to()->isa<Run>()) {
            dst = run->def()->isa_lambda();
        } else if (cur->to()->isa<Hlt>()) {
            cur = continuation(cur);
            continue;
        } else {
            dst = cur->to()->isa_lambda();
        }

        if (dst == nullptr || dst->empty()) {
            cur = postdom(cur);
            continue;
        } else if (dst == end) {
            DLOG("end: %", end);
            return;
        }

        Array<Def> call(cur->size());
        call.front() = dst;
        bool all = true;
        for (size_t i = 1, e = call.size(); i != e; ++i) {
            if (!cur->op(i)->isa<Hlt>())
                call[i] = cur->op(i);
            else
                all = false;
        }

        if (auto cached = find(cache_, call)) {      // check for cached version
            jump_to_cached_call(cur, cached, call);
            DLOG("using cached call: %", cur);
            return;
        } else {                                     // no cached version found... create a new one
            auto dropped = drop(cur, call);

            if (dropped->to() == world().branch()) { // don't peel loops
                if (!scope().inner_contains(dst) || scope().f_cfg().num_preds(dst) != 1) {
                    cur = postdom(cur);
                    continue;
                }
            }

            dirty_ = true;
            cache_[call] = dropped;
            jump_to_cached_call(cur, dropped, call);
            if (all) {
                cur->jump(dropped->to(), dropped->args());
                done_.erase(cur);
            } else
                cur = dropped;
        }
    }
}

Lambda* PartialEvaluator::postdom(Lambda* cur) {
    const auto& postdomtree = scope().b_cfg().domtree();
    if (auto n = scope().cfa(cur)) {
        auto p = postdomtree.idom(n);
        DLOG("postdom: % -> %", n, p);
        return p->lambda();
    }

    WLOG("no postdom found for % at %", cur, cur->loc());
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
        if (auto evalop = Def(primop)->isa<EvalOp>())
            evalop->replace(evalop->def());
    }
}

//------------------------------------------------------------------------------

}
