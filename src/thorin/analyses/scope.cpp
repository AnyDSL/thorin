#include "thorin/analyses/scope.h"

#include <algorithm>
#include <iostream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

uint32_t Scope::sid_counter_ = 1;
uint32_t Scope::candidate_counter_ = 1;

Scope::Scope(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
    , sid_(sid_counter_++)
{
#ifndef NDEBUG
    assert(!entries.empty());
    for (auto entry : entries) assert(!entry->is_proxy());
#endif

    rev_rpo_.push_back(nullptr); // reserve for exit
    identify_scope(entries);
    number(entries);
    build_cfg(entries);
    build_in_scope();
    ++candidate_counter_;
}

Scope::~Scope() {
    for (auto lambda : rpo())
        lambda->unregister(this);

    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        entry()->destroy_body();
    if (exit() != entry() && !exit()->empty() && exit()->to()->isa<Bottom>())
        exit()->destroy_body();
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    for (auto entry : entries) {
        if (!is_candidate(entry)) {
            std::queue<Def> queue;

            auto insert_lambda = [&] (Lambda* lambda) {
                assert(!lambda->is_proxy());
                for (auto param : lambda->params()) {
                    if (!param->is_proxy()) {
                        set_candidate(param);
                        queue.push(param);
                    }
                }

                assert(std::find(rev_rpo_.begin(), rev_rpo_.end(), lambda) == rev_rpo_.end());
                rev_rpo_.push_back(lambda);
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
        }
    }

#ifndef NDEBUG
    for (auto lambda : rev_rpo().slice_from_begin(1))
        assert(is_candidate(lambda));
#endif
}

void Scope::number(ArrayRef<Lambda*> entries) {
    size_t n = 1; // reserve 0 for exit
    if (entries.size() == 1)
        n = number(entries.front(), n);
    else {
        auto entry = world().meta_lambda();
        rev_rpo_.push_back(entry);
        set_candidate(entry);
        for (auto entry : entries) {
            if (!entry->find(this)) // if not visited
                n = number(entry, n);
        }
        n = number(entry, n);
    }

    // remove unreachable lambdas and their params from candidate set
    for (auto lambda : rev_rpo().slice_from_begin(1)) {
        if (lambda->find(this) == nullptr) {
            for (auto param : lambda->params()) {
                if (!param->is_proxy())
                    unset_candidate(param);
            }
            unset_candidate(lambda);
        }
    }

    // remove unreachable lambdas and sort remaining stuff in post-order
    std::sort(rev_rpo_.begin()+1, rev_rpo_.end(), [&](Lambda* l1, Lambda* l2) {
        auto info1 = l1->find(this);
        auto info2 = l2->find(this);
        if (info1 && info2) return info1->rev_rpo_id < info2->rev_rpo_id;
        if (info1) return true;
        if (info2) return false;
        return l1->gid_ < l2->gid_;
    });
    rev_rpo_.resize(n);

    // fill rpo
    assert(rpo_.empty());
    rpo_.resize(n);

    for (size_t i = n; i-- != 1;) {
        auto lambda = rev_rpo_[i];
        auto info = lambda->find(this);
        assert(info && info->rev_rpo_id == i && info->rpo_id == size_t(-1));
        info->rpo_id = n-1 - i;
        rpo_[info->rpo_id] = lambda;
    }
}

size_t Scope::number(Lambda* cur, size_t i) {
    cur->scopes_.emplace_front(this);

    for (auto succ : cur->succs()) {
        if (is_candidate(succ)) {
            if (!succ->find(this)) // if not visited
                i = number(succ, i);
        }
    }

    assert(cur->scopes_.front().scope == this && "front item does not point to this scope");
    assert(cur->scopes_.front().rev_rpo_id == size_t(-1) && "already set");
    cur->scopes_.front().rev_rpo_id = i;
    return i+1;
}

void Scope::build_cfg(ArrayRef<Lambda*> entries) {
    succs_.resize(size());
    preds_.resize(size());
    for (auto lambda : rpo().slice_num_from_end(1)) {
        for (auto succ : lambda->succs()) {
            if (succ->find(this))
                link(lambda, succ);
        }
    }

    // link meta lambda to real entries
    if (entries.size() != 1) {
        for (auto e : entries)
            link(entry(), e);
    }

    // find exits
    LambdaSet entry_set(entries.begin(), entries.end());
    LambdaSet exits;

    for (auto lambda : rpo().slice_num_from_end(1)) {
        for (auto succ : succs(lambda)) {
            if (!entry_set.contains(succ))
                goto next;
        }
        exits.insert(lambda);
next:;
    }

    assert(!exits.empty() && "TODO");
    auto exit = world().meta_lambda();
    exit->scopes_.emplace_front(this);
    set_candidate(exit);
    rev_rpo_.front() = exit;
    rpo_.back() = exit;
    auto& info = exit->scopes_.front();
    info.rpo_id = size()-1;
    info.rev_rpo_id = 0;
    for (auto e : exits)
        link(e, exit);
}

void Scope::build_in_scope() {
    std::queue<Def> queue;
    auto enqueue = [&] (Def def) {
        if (!def->isa_lambda() && is_candidate(def) && !in_scope_.contains(def)) {
            in_scope_.insert(def);
            queue.push(def);
        }
    };

    for (auto lambda : rpo_) {
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

}
