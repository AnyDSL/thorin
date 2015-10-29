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
    void merge_lambdas();
    void eliminate_params();
    void unreachable_code_elimination();
    void dead_code_elimination();
    void verify_closedness();
    void within(const DefNode*);
    void set_live(const PrimOp* primop) { nprimops_.insert(primop); primop->live_ = counter_; }
    void set_reachable(Lambda* lambda)  { nlambdas_.insert(lambda); lambda->reachable_ = counter_; }
    static bool is_live(const PrimOp* primop) { return primop->live_ == counter_; }
    static bool is_reachable(Lambda* lambda) { return lambda->reachable_ == counter_; }

private:
    World& world_;
    LambdaSet nlambdas_;
    World::PrimOps nprimops_;
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
        auto lambda = (*succs.begin())->lambda();
        if (lambda->num_uses() == 1 && lambda == n->lambda()->to())
            return children.front();
    }
    return nullptr;
}

void Merger::merge(const CFNode* /*n*/) {
#if 0
    const DomNode* cur = n;
    if (n->isa<InNode>()) {
        for (const DomNode* next = dom_succ(cur); next != nullptr; cur = next, next = dom_succ(next)) {
            assert(cur->lambda()->num_args() == next->lambda()->num_params());
            for (size_t i = 0, e = cur->lambda()->num_args(); i != e; ++i)
                Def(next->lambda()->param(i))->replace(cur->lambda()->arg(i));
            cur->lambda()->destroy_body();
        }

        if (cur != n)
            n->lambda()->jump(cur->lambda()->to(), cur->lambda()->args());

    }

    for (auto child : cur->children())
        merge(child);
#endif
}

void Cleaner::merge_lambdas() {
    Scope::for_each(world(), [] (const Scope& scope) { Merger merger(scope); });
}

void Cleaner::eliminate_params() {
    for (auto olambda : world().copy_lambdas()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;
        size_t i = 0;
        for (auto param : olambda->params()) {
            if (param->is_proxy())
                proxy_idx.push_back(i++);
            else
                param_idx.push_back(i++);
        }

        if (!proxy_idx.empty()) {
            auto nlambda = world().lambda(world().fn_type(olambda->type()->args().cut(proxy_idx)),
                                            olambda->loc(), olambda->cc(), olambda->intrinsic(), olambda->name);
            size_t j = 0;
            for (auto i : param_idx) {
                olambda->param(i)->replace(nlambda->param(j));
                nlambda->param(j++)->name = olambda->param(i)->name;
            }

            nlambda->jump(olambda->to(), olambda->args());
            olambda->destroy_body();

            for (auto use : olambda->uses()) {
                if (auto ulambda = use->isa_lambda()) {
                    assert(use.index() == 0 && "deleted param of lambda used as argument");
                    ulambda->jump(nlambda, ulambda->args().cut(proxy_idx));
                }
                // else must be a dead 'select' primop
            }
        }
    }
}

void Cleaner::unreachable_code_elimination() {
    std::queue<const Lambda*> queue;
    auto enqueue = [&] (Lambda* lambda) {
        lambda->refresh();
        set_reachable(lambda);
        queue.push(lambda);
    };

    for (auto lambda : world().externals())
        enqueue(lambda);
    enqueue(world().branch());
    enqueue(world().end_scope());

    while (!queue.empty()) {
        auto lambda = pop(queue);
        for (auto succ : lambda->succs()) {
            if (!is_reachable(succ))
                enqueue(succ);
        }
    }

    for (auto lambda : world().lambdas()) {
        if (!is_reachable(lambda))
            lambda->destroy_body();
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

    for (auto lambda : world().lambdas()) {
        for (auto op : lambda->ops()) {
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
    auto check = [&](const DefNode* def) {
        within(def->representative_);
        for (auto op : def->ops())
            within(op.node());
        for (auto use : def->uses_)
            within(use.def().node());
        for (auto r : def->representatives_of_)
            within(r);
    };

    for (auto primop : world().primops())
        check(primop);
    for (auto lambda : world().lambdas()) {
        check(lambda);
        for (auto param : lambda->params())
            check(param);
    }
}

void Cleaner::within(const DefNode* def) {
    //assert(world.types().find(*def->type()) != world.types().end());
    if (auto primop = def->isa<PrimOp>()) {
        assert(world().primops().find(primop) != world().primops().end());
    } else if (auto lambda = def->isa_lambda())
        assert(world().lambdas().find(lambda) != world().lambdas().end());
    else
        within(def->as<Param>()->lambda());
}

void Cleaner::cleanup() {
    //merge_lambdas();
    eliminate_params();
    unreachable_code_elimination();
    dead_code_elimination();

    // unlink dead primops from the rest
    for (auto primop : world().primops()) {
        if (!is_live(primop)) {
            primop->unregister_uses();
            primop->unlink_representative();
        }
    }

    // unlink unreachable lambdas from the rest
    for (auto lambda : world().lambdas()) {
        if (!is_reachable(lambda)) {
            for (auto param : lambda->params())
                param->unlink_representative();
            lambda->unlink_representative();
        }
    }

    swap(world().primops_, nprimops_);
    swap(world().lambdas_, nlambdas_);
#ifndef NDEBUG
    verify_closedness();
#endif

    // delete dead primops
    for (auto primop : nprimops_) {
        if (!is_live(primop))
            delete primop;
    }

    // delete unreachable lambdas
    for (auto lambda : nlambdas_) {
        if (!is_reachable(lambda))
            delete lambda;
    }

#ifndef NDEBUG
    for (auto primop : world().primops())
        assert(!primop->is_outdated());
#endif

    debug_verify(world());
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
