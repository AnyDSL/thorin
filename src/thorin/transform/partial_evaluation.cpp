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

    Scope& scope() { return scope_; }
    World& world() { return scope().world(); }
    void run();
    void eval(Lambda* begin, Lambda* end);
    void enqueue(Lambda* lambda) {
        if (scope().outer_contains(lambda) && !visit(visited_, lambda))
            queue_.push(lambda);
    }

private:
    Scope& scope_;
    LambdaSet done_;
    std::queue<Lambda*> queue_;
    LambdaSet visited_;
    HashMap<Array<Def>, Lambda*> cache_;
};

static Lambda* continuation(Lambda* lambda) {
    return lambda->num_args() != 0 ? lambda->args().back()->isa_lambda() : (Lambda*) nullptr;
}

void PartialEvaluator::run() {
    enqueue(scope().entry());

    while (!queue_.empty()) {
        auto lambda = pop(queue_);

        if (lambda->to()->isa<Run>()) {
            eval(lambda, continuation(lambda));
            scope_.update();
        }

        for (auto succ : scope().f_cfg().succs(lambda))
            enqueue(succ->lambda());
    }
}

void PartialEvaluator::eval(Lambda* cur, Lambda* end) {
    if (end == nullptr)
        WLOG("no matching end: % at %", cur->unique_name(), cur->loc());
    else
        DLOG("eval: % -> %", cur->unique_name(), end->unique_name());

    while (true) {
        if (cur == nullptr) {
            DLOG("cur is nullptr");
            return;
        }
        if (done_.contains(cur)) {
            DLOG("already done: %", cur->unique_name());
            return;
        }
        if (cur->empty()) {
            DLOG("empty: %", cur->unique_name());
            return;
        }

        Lambda* dst = nullptr;
        if (auto run = cur->to()->isa<Run>()) {
            dst = run->def()->isa_lambda();
        } else if (cur->to()->isa<Hlt>()) {
            auto& s = scope_.update();
            assert(s.outer_contains(cur));
            for (auto succ : s.f_cfg().succs(cur))
                enqueue(succ->lambda());
            cur = continuation(cur);
            continue;
        } else {
            dst = cur->to()->isa_lambda();
        }

        if (dst == nullptr) {
            DLOG("dst is nullptr; cur: %", cur->unique_name());
            return;
        }

        if (dst == end) {
            DLOG("end: %", end->unique_name());
            return;
        }

        done_.insert(cur);

        if (dst->empty()) {
            auto& postdomtree = scope_.update().b_cfg().domtree();
            if (auto n = scope().cfa(cur)) {
                auto p = postdomtree.idom(n);
                DLOG("postdom: % -> %", n, p);
                cur = p->lambda();
                continue;
            }
            WLOG("no postdom found for % at %", cur->unique_name(), cur->loc());
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

        if (auto cached = find(cache_, call)) { // check for cached version
            jump_to_cached_call(cur, cached, call);
            DLOG("using cached call: %", cur->unique_name());
            return;
        } else {                                // no cached version found... create a new one
            auto dropped = drop(cur, call);
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
