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

    // remove unreachable lambdas and their params from candidate set and unregister from this scope
    for (auto lambda : po_) {
        if (lambda->find(this) == nullptr) {
            for (auto param : lambda->params()) {
                if (!param->is_proxy())
                    unset_candidate(param);
            }
            unset_candidate(lambda);
        }
    }

    // remove unreachable lambdas and sort remaining stuff in post-order
    std::sort(po_.begin(), po_.end(), [&](Lambda* l1, Lambda* l2) {
        auto info1 = l1->find(this);
        auto info2 = l2->find(this);
        if (info1 && info2) return info1->po_index < info2->po_index;
        if (info1) return true;
        if (info2) return false;
        return l1->gid_ < l2->gid_;
    });
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
    succs_.resize(size() + 1); // alloc one more:
    preds_.resize(size() + 1); // we could need a meta lambda as exit
    for (auto lambda : rpo_) {
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

    for (auto lambda : rpo()) {
        for (auto succ : succs(lambda)) {
            if (!entry_set.contains(succ))
                goto next;
        }
        exits.insert(lambda);
next:;
    }

    //if (exits.size() == 1) {
        //auto exit = *exits.begin();
        //auto  pox =  po_index(exit);
        //auto rpox = rpo_index(exit);
        //std::swap(*exit->find(this), *rpo_.back()->find(this));
        //std::swap( po_[ pox],  po_.front());
        //std::swap(rpo_[rpox], rpo_.back());
        //std::swap(succs_[rpox], succs_.back());
        //std::swap(preds_[rpox], preds_.back());
    //} else {
        assert(!exits.empty() && "TODO");
        auto exit = world().meta_lambda();
        exit->scopes_.emplace_front(this);
        set_candidate(exit);
        po_.push_back(nullptr);
        rpo_.push_back(exit);
        auto& info = exit->scopes_.front();
        info.po_index = 0;
        info.rpo_index = size()-1;
        // correct po_ lambdas
        for (size_t i = size(); i-- != 1;) {
            po_[i] = po_[i-1];
            ++po_[i]->find(this)->po_index;
            assert(po_[i]->find(this)->po_index == i);
        }
        po_[0] = exit;

        for (auto e : exits)
            link(e, exit);
    //}
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
