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

uint32_t Scope::counter_ = 0;

Scope::Scope(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
{
    assert(!entries.empty());
#ifndef NDEBUG
    for (auto entry : entries)
        assert(!entry->is_proxy());
#endif

    identify_scope(entries);

    size_t n;
    if (entries.size() == 1) {
        n = number(entries.front(), 0);
    } else {
        auto entry = world.meta_lambda();
        po_.push_back(entry);
        set_candidate(entry);

        n = number(entry, 0);
        assert(n == 1);
        for (auto entry : entries) {
            if (!entry->find(this)) // if not visited
                n = number(entry, n);
        }
    }

    // sort in post-order
    std::sort(po_.begin(), po_.end(), [&](Lambda* l1, Lambda* l2) { return po_index(l1) < po_index(l2); });

    // remove unreachable lambdas and their params from candidate set
    std::for_each(po_.begin() + n, po_.end(), [&] (Lambda* lambda) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                unset_candidate(param);
        }
        unset_candidate(lambda);
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

    build_cfg(entries);
    //auto exit = find_exit();
    //link_exit(entry, exit);
    //ScopeView(*this,  true).rpo_numbering(entry);
    //ScopeView(*this, false).rpo_numbering(exit);
}

Scope::~Scope() {
    ++counter_;

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
                    if (!in_scope_.contains(use)) {
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

#if 0
Lambda* Scope::find_exit() {
    LambdaSet exits;

    for (auto lambda : rpo_) {
        if (succs_[lambda].empty())
            exits.insert(lambda);
    }

    Lambda* exit;
    if (exits.size() == 1)
        exit = *exits.begin();
    else {
        exit = world().meta_lambda();
        rpo_.push_back(exit);
        po_.push_back(exit);
        in_scope_.insert(exit);
        for (auto e : exits) {
            link_succ(e, exit);
            link_pred(e, exit);
        }
    }

    return exit;
}

void Scope::link_exit(Lambda* entry, Lambda* exit) {
    LambdaSet done;
    auto r = ScopeView(*this, false).reachable(exit);
    link_exit(done, r, entry, exit);
}

void Scope::link_exit(LambdaSet& done, LambdaSet& reachable, Lambda* cur, Lambda* exit) {
    for (auto succ : succs_[cur]) {
        if (!visit(done, succ))
            link_exit(done, reachable, succ, exit);
    }

    if (!reachable.contains(cur)) {
        bool still_unreachable = true;
        for (auto succ : succs_[cur]) {
            if (reachable.contains(succ)) {
                still_unreachable = false;
                break;
            }
        }
        reachable.insert(cur);
        if (still_unreachable) {
            link_succ(cur, exit);
            link_pred(cur, exit);
        }
    }
}

#endif

}
