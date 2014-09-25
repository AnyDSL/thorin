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

uint32_t Scope::counter_ = 1;

Scope::Scope(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
{
#ifndef NDEBUG
    assert(!entries.empty());
    for (auto entry : entries) assert(!entry->is_proxy());
#endif

    identify_scope(entries);
    number(entries);
    build_cfg(entries);
    build_in_scope();
}

Scope::~Scope() {
    for (auto lambda : rpo())
        lambda->unregister(this);

    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        entry()->destroy_body();
    if (exit() != entry() && !exit()->empty() && exit()->to()->isa<Bottom>())
        exit()->destroy_body();

    ++counter_;
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

                assert(std::find(po_.begin(), po_.end(), lambda) == po_.end());
                po_.push_back(lambda);
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
    for (auto lambda : po_)
        assert(is_candidate(lambda));
#endif
}

void Scope::number(ArrayRef<Lambda*> entries) {
    size_t n;
    if (entries.size() == 1)
        n = number(entries.front(), 0);
    else {
        auto entry = world().meta_lambda();
        po_.push_back(entry);
        set_candidate(entry);
        n = 0;
        for (auto entry : entries) {
            if (!entry->find(this)) // if not visited
                n = number(entry, n);
        }
        n = number(entry, n);
    }
    assert(po_.size() == n);

    // sort in post-order
    std::sort(po_.begin(), po_.end(), [&](Lambda* l1, Lambda* l2) { return po_index(l1) < po_index(l2); });

    // remove unreachable lambdas and their params from candidate set and unregister from this scope
    std::for_each(po_.begin() + n, po_.end(), [&] (Lambda* lambda) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                unset_candidate(param);
        }
        unset_candidate(lambda);
        lambda->unregister(this);
    });

    // remove unreachable lambdas
    po_.resize(n);

    // fill rpo
    assert(rpo_.empty());
    rpo_.resize(n);

    for (size_t i = n; i-- != 0;) {
        auto lambda = po_[i];
        auto info = lambda->find(this);
        assert(info && info->po_index == i && info->rpo_index == size_t(-1));
        info->rpo_index = n-1 - i;
        rpo_[info->rpo_index] = lambda;
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

    assert(cur->scopes_.front().po_index == size_t(-1) && "already set");
    assert(cur->scopes_.front().scope == this && "front item does not point to this scope");
    cur->scopes_.front().po_index = i;
    return i+1;
}

void Scope::build_cfg(ArrayRef<Lambda*> entries) {
    succs_.resize(size());
    preds_.resize(size());
    for (auto lambda : rpo_) {
        for (auto succ : lambda->succs()) {
            if (is_candidate(succ))
                link(lambda, succ);
        }
    }

    // link meta lambda to real entries
    if (entries.size() != 1) {
        for (auto e : entries)
            link(entry(), e);
    }
}

void Scope::find_exits(ArrayRef<Lambda*> entries) {
    LambdaSet entry_set(entries.begin(), entries.end());
    LambdaSet exits;

    for (auto lambda : rpo()) {
        for (auto succ : succs(lambda)) {
            if (!entry_set.contains(succ))
                goto next;
        }
        exits.insert(lambda);
next:;
    }

    if (exits.size() == 1) {
        auto exit = *exits.begin();
        auto  pox =  po_index(exit);
        auto rpox = rpo_index(exit);
        std::swap(*exit->find(this), *rpo_.back()->find(this));
        std::swap( po_[ pox],  po_.front());
        std::swap(rpo_[rpox], rpo_.back());
    } else {
        auto exit = world().meta_lambda();
        rpo_.push_back(exit);
        assert(false && "TODO");
        //set_candidate(exit);
        //po_.push_front(exit);
        //in_scope_.insert(exit);
        for (auto e : exits)
            link(e, exit);
    }
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
        for (auto param : lambda->params())
            in_scope_.insert(param);
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
