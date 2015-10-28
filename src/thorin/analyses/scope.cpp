#include "thorin/analyses/scope.h"

#include <algorithm>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/util/queue.h"

namespace thorin {

uint32_t Scope::id_counter_ = 1;
uint32_t Scope::candidate_counter_ = 1;

Scope::Scope(Lambda* entry)
    : world_(entry->world())
    , id_(id_counter_++)
{
    run(entry);
}

Scope::~Scope() { cleanup(); }

const Scope& Scope::update() {
    cleanup();
    auto e = entry();
    lambdas_.clear();
    in_scope_.clear();
    cfa_.release();
    id_ = id_counter_++;
    run(e);
    return *this;
}

void Scope::run(Lambda* entry) {
    assert(!entry->is_proxy());
    identify_scope(entry);
    build_in_scope();
    ++candidate_counter_;
    verify();
}

void Scope::cleanup() {
    for (auto lambda : lambdas())
        lambda->unregister_scope(this);
}

void Scope::identify_scope(Lambda* entry) {
    std::queue<Def> queue;
    assert(!is_candidate(entry));

    auto insert_lambda = [&] (Lambda* lambda) {
        assert(!lambda->is_proxy());
        for (auto param : lambda->params()) {
            if (!param->is_proxy()) {
                set_candidate(param);
                queue.push(param);
            }
        }

        assert(std::find(lambdas_.begin(), lambdas_.end(), lambda) == lambdas_.end());
        lambdas_.push_back(lambda);
    };

    insert_lambda(entry);
    set_candidate(entry);

    while (!queue.empty()) {
        auto def = pop(queue);
        for (auto use : def->uses()) {
            if (!is_candidate(use)) {
                if (auto ulambda = use->isa_lambda())
                    insert_lambda(ulambda);
                set_candidate(use);
                queue.push(use);
            }
        }
    }

    lambdas_.push_back(world().end_scope());
    set_candidate(world().end_scope());

    for (size_t i = 0, e = size(); i != e; ++i) {
        auto lambda = lambdas_[i];
        lambda->register_scope(this)->index = i;
        assert(is_candidate(lambda));
    }
    assert(lambdas().front() == entry);
}

void Scope::verify() const {
#ifndef NDEBUG
    for (auto lambda : lambdas_) {
        auto info = lambda->find_scope(this);
        assert(info->scope == this);
        assert((*this)[info->index] == lambda);
    }
#endif
}

void Scope::build_in_scope() {
    std::queue<Def> queue;
    auto enqueue = [&] (Def def) {
        if (!def->isa_lambda() && is_candidate(def) && !in_scope_.contains(def)) {
            in_scope_.insert(def);
            queue.push(def);
        }
    };

    for (auto lambda : lambdas()) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                in_scope_.insert(param);
        }
        in_scope_.insert(lambda);

        for (auto op : lambda->ops())
            enqueue(op);

        while (!queue.empty()) {
            for (auto op : pop(queue)->ops())
                enqueue(op);
        }
    }
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const CFNode* Scope::cfa(Lambda* lambda) const { return cfa()[lambda]; }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(Scope&)> f) {
    LambdaSet done;
    std::queue<Lambda*> queue;

    for (auto lambda : world.externals()) {
        assert(!lambda->empty() && "external must not be empty");
        done.insert(lambda);
        queue.push(lambda);
    }

    while (!queue.empty()) {
        auto lambda = pop(queue);
        if (elide_empty && lambda->empty())
            continue;
        Scope scope(lambda);
        f(scope);
        for (auto lambda : scope)
            done.insert(lambda);

        for (auto lambda : scope) {
            for (auto succ : lambda->succs()) {
                if (!done.contains(succ)) {
                    done.insert(succ);
                    queue.push(succ);
                }
            }
        }
    }
}

template void Scope::for_each<true> (const World&, std::function<void(Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(Scope&)>);

}
