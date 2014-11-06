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
    assert(!entry->is_proxy());
    identify_scope(entry);
    build_in_scope();
    ++candidate_counter_;
}

Scope::~Scope() {
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

    for (size_t i = 0, e = size(); i != e; ++i) {
        auto lambda = lambdas_[i];
        lambda->register_scope(this)->sid = i;
        assert(is_candidate(lambda));
    }
    assert(lambdas().front() == entry);
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

const CFG* Scope::cfg() const { return lazy_init(this, cfg_); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(const Scope&)> f) {
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

template void Scope::for_each<true> (const World&, std::function<void(const Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(const Scope&)>);

}
