#include "thorin/world.h"
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

void Cleaner::merge_lambdas() {
    for (auto src : world().lambdas()) {
        if (auto dst = src->to()->isa_lambda()) {
            if (src != dst && !dst->empty() && !world().is_external(dst) && dst->num_uses() == 1) {
                for (size_t i = 0, e = src->num_args(); i != e; ++i)
                    dst->param(i)->replace(src->arg(i));

                src->jump(dst->to(), dst->args());
                dst->destroy_body();
            }
        }
    }
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
                                            olambda->cc(), olambda->intrinsic(), olambda->name);
            size_t j = 0;
            for (auto i : param_idx) {
                olambda->param(i)->replace(nlambda->param(j));
                nlambda->param(j++)->name = olambda->param(i)->name;
            }

            nlambda->jump(olambda->to(), olambda->args());
            olambda->destroy_body();

            for (auto use : olambda->uses()) {
                auto ulambda = use->as_lambda();
                assert(use.index() == 0 && "deleted param of lambda used as argument");
                ulambda->jump(nlambda, ulambda->args().cut(proxy_idx));
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
            within(r); };

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
    merge_lambdas();
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
    for (auto primop : world().primops()) {
        if (!is_live(primop))
            delete primop;
    }

    // delete unreachable lambdas
    for (auto lambda : world().lambdas()) {
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
