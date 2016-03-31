#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/queue.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}
    ~Cleaner() { ++counter_; }

    World& world() { return world_; }
    void cleanup();
    void merge_continuations();
    void eliminate_params();
    void unreachable_code_elimination();
    void dead_code_elimination();
    void verify_closedness();
    void within(const Def*);
    void set_live(const PrimOp* primop) { nprimops_.insert(primop); primop->live_ = counter_; }
    void set_reachable(Continuation* continuation)  { ncontinuations_.insert(continuation); continuation->reachable_ = counter_; }
    static bool is_live(const PrimOp* primop) { return primop->live_ == counter_; }
    static bool is_reachable(Continuation* continuation) { return continuation->reachable_ == counter_; }

private:
    World& world_;
    ContinuationSet ncontinuations_;
    World::PrimOpSet nprimops_;
    Def2Def old2new_;
    static uint32_t counter_;
};

uint32_t Cleaner::counter_ = 1;

class Merger {
public:
    Merger(const Scope& scope)
        : scope(scope)
        , cfg(scope.f_cfg())
        , domtree(cfg.domtree())
    {
        merge(domtree.root());
    }

    void merge(const CFNode* n);
    const CFNode* dom_succ(const CFNode* n);
    World& world() { return scope.world(); }

    const Scope& scope;
    const F_CFG& cfg;
    const DomTree& domtree;
};

const CFNode* Merger::dom_succ(const CFNode* n) {
    const auto& succs = cfg.succs(n);
    const auto& children = domtree.children(n);
    if (succs.size() == 1 && children.size() == 1 && *succs.begin() == (*children.begin())) {
        auto continuation = (*succs.begin())->continuation();
        if (continuation->num_uses() == 1 && continuation == n->continuation()->callee())
            return children.front();
    }
    return nullptr;
}

void Merger::merge(const CFNode* n) {
    auto cur = n;
    for (auto next = dom_succ(cur); next != nullptr; cur = next, next = dom_succ(next)) {
        assert(cur->continuation()->num_args() == next->continuation()->num_params());
        for (size_t i = 0, e = cur->continuation()->num_args(); i != e; ++i)
            next->continuation()->param(i)->replace(cur->continuation()->arg(i));
        cur->continuation()->destroy_body();
    }

    if (cur != n)
        n->continuation()->jump(cur->continuation()->callee(), cur->continuation()->type_args(), cur->continuation()->args(), cur->continuation()->jump_loc());

    for (auto child : domtree.children(cur))
        merge(child);
}

void Cleaner::merge_continuations() {
    Scope::for_each(world(), [] (const Scope& scope) { Merger merger(scope); });
}

void Cleaner::eliminate_params() {
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (!ocontinuation->empty() && !world().is_external(ocontinuation)) {
            for (auto use : ocontinuation->uses()) {
                if (use.index() != 0 || !use->isa_continuation())
                    goto next_continuation;
            }

            for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
                auto param = ocontinuation->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty() && ocontinuation->num_type_params() == 0) { // TODO do this for polymorphic functions, too
                auto ncontinuation = world().continuation(world().fn_type(ocontinuation->type()->args().cut(proxy_idx)),
                                            ocontinuation->loc(), ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->name);
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace(ncontinuation->param(j));
                    ncontinuation->param(j++)->name = ocontinuation->param(i)->name;
                }

                ncontinuation->jump(ocontinuation->callee(), ocontinuation->type_args(), ocontinuation->args(), ocontinuation->jump_loc());
                ocontinuation->destroy_body();

                for (auto use : ocontinuation->uses()) {
                    auto ucontinuation = use->as_continuation();
                    assert(use.index() == 0);
                    ucontinuation->jump(ncontinuation, ucontinuation->type_args(), ucontinuation->args().cut(proxy_idx), ucontinuation->jump_loc());
                }
            }
        }
next_continuation:;
    }
}

void Cleaner::unreachable_code_elimination() {
    std::queue<const Continuation*> queue;
    auto enqueue = [&] (Continuation* continuation) {
        continuation->refresh(old2new_);
        set_reachable(continuation);
        queue.push(continuation);
    };

    for (auto continuation : world().externals())
        enqueue(continuation);
    enqueue(world().branch());
    enqueue(world().end_scope());

    while (!queue.empty()) {
        auto continuation = pop(queue);
        for (auto succ : continuation->succs()) {
            if (!is_reachable(succ))
                enqueue(succ);
        }
    }

    for (auto continuation : world().continuations()) {
        if (!is_reachable(continuation))
            continuation->destroy_body();
    }
}

void Cleaner::dead_code_elimination() {
    std::queue<const PrimOp*> queue;
    auto enqueue = [&] (const PrimOp* primop) {
        if (!is_live(primop)) {
            set_live(primop);
            queue.push(primop);
        }
    };

    for (auto continuation : world().continuations()) {
        for (auto op : continuation->ops()) {
            if (auto primop = op->isa<PrimOp>())
                enqueue(primop);
        }
    }

    while (!queue.empty()) {
        auto primop = pop(queue);
        for (auto op : primop->ops()) {
            if (auto primop = op->isa<PrimOp>())
                enqueue(primop);
        }
    }
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert(op->uses_.find(Use(i++, def)) != op->uses_.end() && "can't find def in op's uses");
        }

        for (auto use : def->uses_) {
            within(use.def());
            assert(use->op(use.index()) == def && "can't use doesn't point to def");
        }
    };

    for (auto primop : world().primops())
        check(primop);
    for (auto continuation : world().continuations()) {
        check(continuation);
        for (auto param : continuation->params())
            check(param);
    }
}

void Cleaner::within(const Def* def) {
    //assert(world.types().find(*def->type()) != world.types().end());
    if (auto primop = def->isa<PrimOp>()) {
        assert_unused(world().primops().find(primop) != world().primops().end());
    } else if (auto continuation = def->isa_continuation())
        assert_unused(world().continuations().find(continuation) != world().continuations().end());
    else
        within(def->as<Param>()->continuation());
}

void Cleaner::cleanup() {
#ifndef NDEBUG
    for (const auto& p : world().trackers_)
        assert(p.second.empty() && "there are still live trackers before running cleanup");
#endif

    merge_continuations();
    eliminate_params();
    unreachable_code_elimination();
    dead_code_elimination();

    for (auto primop : world().primops()) {
        if (!is_live(primop))
            primop->unregister_uses();
    }

    swap(world().primops_, nprimops_);
    swap(world().continuations_, ncontinuations_);
#ifndef NDEBUG
    verify_closedness();
#endif

    for (auto primop : nprimops_) {
        if (!is_live(primop))
            delete primop;
    }

    for (auto continuation : ncontinuations_) {
        if (!is_reachable(continuation))
            delete continuation;
    }

#ifndef NDEBUG
    for (auto primop : world().primops())
        assert(!primop->is_outdated());

    for (const auto& p : world().trackers_)
        assert(p.second.empty() && "trackers needed during cleanup phase");
#endif
    world().trackers_.clear();

    debug_verify(world());
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
