#include "thorin/world.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/queue.h"
#include "thorin/be/thorin.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void cleanup();
    void eliminate_params();
    void unreachable_code_elimination();
    void dead_code_elimination();

private:
    World& world_;
    LambdaSet reachable_;
    World::PrimOps live_;
};

void Cleaner::eliminate_params() {
    for (auto olambda : world().copy_lambdas()) {
        if (olambda->empty())
            continue;

        olambda->clear();
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;
        size_t i = 0;
        for (auto param : olambda->params()) {
            if (param->is_proxy())
                proxy_idx.push_back(i++);
            else
                param_idx.push_back(i++);
        }

        if (proxy_idx.empty())
            continue;

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

void Cleaner::unreachable_code_elimination() {
    std::queue<const Lambda*> queue;
    auto enqueue = [&] (Lambda* lambda) {
        for (size_t i = 0, e = lambda->size(); i != e; ++i)
            lambda->op(i); // HACK: refresh

        reachable_.insert(lambda);
        queue.push(lambda);
    };

    for (auto lambda : world().externals()) {
        assert(!reachable_.contains(lambda));
        enqueue(lambda);
    }

    while (!queue.empty()) {
        auto lambda = pop(queue);
        for (auto succ : lambda->succs()) {
            if (!reachable_.contains(succ))
                enqueue(succ);
        }
    }

    for (auto lambda : world().lambdas()) {
        if (!reachable_.contains(lambda))
            lambda->destroy_body();
    }
}

void Cleaner::dead_code_elimination() {
    std::queue<const PrimOp*> queue;
    auto enqueue = [&] (const PrimOp* primop) {
        if (!live_.contains(primop)) {
            live_.insert(primop);
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

void Cleaner::cleanup() {
    eliminate_params();
    unreachable_code_elimination();
    dead_code_elimination();

    auto unlink_representative = [&] (const DefNode* def) {
        if (def->is_proxy()) {
            auto num = def->representative_->representatives_of_.erase(def);
            assert(num == 1);
        }
    };

    // unlink dead primops from the rest
    for (auto primop : world().primops()) {
        if (!live_.contains(primop)) {
            for (size_t i = 0, e = primop->size(); i != e; ++i)
                primop->unregister_use(i);
            unlink_representative(primop);
        }
    }

    // unlink unreachable lambdas from the rest
    for (auto lambda : world().lambdas()) {
        if (!reachable_.contains(lambda)) {
            for (auto param : lambda->params())
                unlink_representative(param);
            unlink_representative(lambda);
        }
    }

    swap(world().primops_, live_);
    swap(world().lambdas_, reachable_);

    verify_closedness(world());

    // delete dead primops
    for (auto primop : world().primops()) {
        if (!live_.contains(primop))
            delete primop;
    }

    // delete unreachable lambdas
    for (auto lambda : world().lambdas()) {
        if (!reachable_.contains(lambda))
            delete lambda;
    }

#ifndef NDEBUG
    for (auto primop : world().primops())
        assert(primop->up_to_date_);
#endif

    debug_verify(world());
}

void cleanup_world(World& world) {
    emit_thorin(world, true);
    Cleaner(world).cleanup();
}

}
