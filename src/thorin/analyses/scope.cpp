#include "thorin/analyses/scope.h"

#include <algorithm>
#include <fstream>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

Scope::Scope(Lam* entry)
    : world_(entry->world())
    , entry_(entry)
    , exit_(world().end_scope())
{
    run();
}

Scope::~Scope() {}

Scope& Scope::update() {
    defs_.clear();
    free_        = nullptr;
    free_params_ = nullptr;
    cfa_         = nullptr;
    run();
    return *this;
}

void Scope::run() {
    std::queue<const Def*> queue;

    auto enqueue = [&] (const Def* def) {
        if (defs_.insert(def).second) {
            queue.push(def);

            if (auto continuation = def->isa_nom<Lam>()) {
                // when a lambda is part of this scope, we also enqueue its params, and we assert those to be unique
                // TODO most likely redundant once params have the lambda in their ops
                for (auto param : continuation->params()) {
                    auto p = defs_.insert(param);
                    assert_unused(p.second);
                    queue.push(param);
                }
            }
        }
    };

    enqueue(entry_);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (def != entry_) {
            for (auto use : def->uses())
                enqueue(use);
        }
    }

    enqueue(exit_);
}

const DefSet& Scope::free() const {
    if (!free_) {
        free_ = std::make_unique<DefSet>();

        for (auto def : defs_) {
            for (auto op : def->ops()) {
                if (!contains(op))
                    free_->emplace(op);
            }
        }
    }

    return *free_;
}

const ParamSet& Scope::free_params() const {
    if (!free_params_) {
        free_params_ = std::make_unique<ParamSet>();
        unique_queue<DefSet> queue;

        auto enqueue = [&](const Def* def) {
            if (auto param = def->isa<Param>(); param && !param->lambda()->dead_)
                free_params_->emplace(param);
            else if (def->isa<Lam>())
                return;
            else
                queue.push(def);
        };

        for (auto def : free())
            enqueue(def);
        while (!queue.empty()) {
            for (auto op : queue.pop()->ops())
                enqueue(op);
        }
    }

    return *free_params_;
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(Scope&)> f) {
    unique_queue<LamSet> continuation_queue;

    for (auto&& [_, cont] : world.externals()) {
        if (cont->has_body()) continuation_queue.push(cont);
    }

    while (!continuation_queue.empty()) {
        auto continuation = continuation_queue.pop();
        if (elide_empty && !continuation->has_body())
            continue;
        Scope scope(continuation);
        f(scope);

        unique_queue<DefSet> def_queue;
        for (auto def : scope.free())
            def_queue.push(def);

        while (!def_queue.empty()) {
            auto def = def_queue.pop();
            if (auto continuation = def->isa_nom<Lam>())
                continuation_queue.push(continuation);
            else {
                for (auto op : def->ops())
                    def_queue.push(op);
            }
        }
    }
}

template void Scope::for_each<true> (const World&, std::function<void(Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(Scope&)>);

}
