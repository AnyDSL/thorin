#include "thorin/analyses/scope.h"

#include <algorithm>
#include <fstream>

#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/util/queue.h"

namespace thorin {

uint32_t Scope::id_counter_ = 1;
uint32_t Scope::candidate_counter_ = 1;

Scope::Scope(Continuation* entry)
    : world_(entry->world())
    , id_(id_counter_++)
{
    run(entry);
}

Scope::~Scope() { cleanup(); }

void Scope::run(Continuation* entry) {
    identify_scope(entry);
    build_defs();
    ++candidate_counter_;
    verify();
}

void Scope::cleanup() {
    for (auto continuation : continuations())
        continuation->unregister_scope(this);
}

const Scope& Scope::update() {
    cleanup();
    auto e = entry();
    continuations_.clear();
    defs_.clear();
    cfa_.release();
    id_ = id_counter_++;
    run(e);
    return *this;
}

void Scope::identify_scope(Continuation* entry) {
    std::queue<const Def*> queue;
    assert(!is_candidate(entry));

    auto insert_continuation = [&] (Continuation* continuation) {
        for (auto param : continuation->params()) {
            set_candidate(param);
            queue.push(param);
        }

        assert(std::find(continuations_.begin(), continuations_.end(), continuation) == continuations_.end());
        continuations_.push_back(continuation);
    };

    insert_continuation(entry);
    set_candidate(entry);

    while (!queue.empty()) {
        auto def = pop(queue);
        for (auto use : def->uses()) {
            if (!is_candidate(use)) {
                if (auto ucontinuation = use->isa_continuation())
                    insert_continuation(ucontinuation);
                set_candidate(use);
                queue.push(use);
            }
        }
    }

    continuations_.push_back(world().end_scope());
    set_candidate(world().end_scope());

    for (size_t i = 0, e = size(); i != e; ++i) {
        auto continuation = continuations_[i];
        continuation->register_scope(this)->index = i;
        assert(is_candidate(continuation));
    }
    assert(continuations().front() == entry);
}

void Scope::verify() const {
#ifndef NDEBUG
    for (auto continuation : continuations_) {
        auto info = continuation->find_scope(this);
        assert(info->scope == this);
        assert((*this)[info->index] == continuation);
    }
#endif
}

void Scope::build_defs() {
    std::queue<const Def*> queue;
    auto enqueue = [&] (const Def* def) {
        if (!def->isa_continuation() && is_candidate(def) && !defs_.contains(def)) {
            defs_.insert(def);
            queue.push(def);
        }
    };

    for (auto continuation : continuations()) {
        for (auto param : continuation->params())
            defs_.insert(param);
        defs_.insert(continuation);

        for (auto op : continuation->ops())
            enqueue(op);

        while (!queue.empty()) {
            for (auto op : pop(queue)->ops())
                enqueue(op);
        }
    }
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const CFNode* Scope::cfa(Continuation* continuation) const { return cfa()[continuation]; }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(Scope&)> f) {
    ContinuationSet done;
    std::queue<Continuation*> queue;

    auto enqueue = [&] (Continuation* continuation) {
        const auto& p = done.insert(continuation);
        if (p.second)
            queue.push(continuation);
    };

    for (auto continuation : world.externals()) {
        assert(!continuation->empty() && "external must not be empty");
        enqueue(continuation);
    }

    while (!queue.empty()) {
        auto continuation = pop(queue);
        if (elide_empty && continuation->empty())
            continue;
        Scope scope(continuation);
        f(scope);

        for (auto n : scope.f_cfg().reverse_post_order()) {
            for (auto succ : n->continuation()->succs()) {
                if (!scope.contains(succ))
                    enqueue(succ);
            }
        }
    }
}

template void Scope::for_each<true> (const World&, std::function<void(Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(Scope&)>);

std::ostream& Scope::stream(std::ostream& os) const { return schedule(*this).stream(os); }
void Scope::write_thorin(const char* filename) const { return schedule(*this).write_thorin(filename); }
void Scope::thorin() const { return schedule(*this).thorin(); }

}
