#include "thorin/analyses/scope.h"

#include <algorithm>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/util/queue.h"

namespace thorin {

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

    Lambda* entry;
    if (entries.size() == 1)
        entry = entries.front();
    else {
        entry = world.meta_lambda();
        rpo_.push_back(entry);
        set_candidate(entry);
    }
    identify_scope(entries);
    number(entries);
    build_cfg(entries);
    build_rev_rpo(entries);
    build_in_scope();
    ++candidate_counter_;
}

Scope::~Scope() {
    for (auto lambda : rpo())
        lambda->unregister_scope(this);

    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        entry()->destroy_body();
    if (exit() != entry() && !exit()->empty() && exit()->to()->isa<Bottom>())
        exit()->destroy_body();
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    std::queue<Def> queue;
    for (auto entry : entries) {
        if (!is_candidate(entry)) {
            auto insert_lambda = [&] (Lambda* lambda) {
                assert(!lambda->is_proxy());
                for (auto param : lambda->params()) {
                    if (!param->is_proxy()) {
                        set_candidate(param);
                        queue.push(param);
                    }
                }

                assert(std::find(rpo_.begin(), rpo_.end(), lambda) == rpo_.end());
                rpo_.push_back(lambda);
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
    for (auto lambda : rpo())
        assert(is_candidate(lambda));
#endif
}

void Scope::number(ArrayRef<Lambda*> entries) {
    size_t n = 0;
    if (entries.size() == 1)
        n = number(entries.front(), n);
    else {
        for (auto entry : entries) {
            if (!entry->find_scope(this)) // if not visited
                n = number(entry, n);
        }
        n = number(entry(), n);
    }

    // sort in reverse post-order
    std::sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) {
        auto info1 = l1->find_scope(this);
        auto info2 = l2->find_scope(this);
        if (info1 && info2) return info1->rpo_id > info2->rpo_id;
        if (info1) return true;
        if (info2) return false;
        return l1->gid_ < l2->gid_;
    });

    // remove unreachable lambdas and their params from candidate set
    for (auto lambda : rpo().slice_from_begin(n)) {
        assert(lambda->find_scope(this) == nullptr);
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                unset_candidate(param);
        }
        unset_candidate(lambda);
    }

    // remove unreachable lambdas
    rpo_.resize(n);

    // convert post-order numbers to reverse post-order numbers
    for (auto lambda : rpo()) {
        auto info = lambda->find_scope(this);
        info->rpo_id = n-1 - info->rpo_id;
    }
}

size_t Scope::number(Lambda* cur, size_t i) {
    auto info = cur->register_scope(this);

    for (auto succ : cur->succs()) {
        if (is_candidate(succ)) {
            if (!succ->find_scope(this)) // if not visited
                i = number(succ, i);
        }
    }

    assert(cur->scopes_.front().scope == this && "front item does not point to this scope");
    assert(cur->scopes_.front().rpo_id == size_t(-1) && "already set");
    return (info->rpo_id = i) + 1;
}

void Scope::build_cfg(ArrayRef<Lambda*> entries) {
    succs_.resize(size() + 1); // maybe we need one extra alloc for fake exit
    preds_.resize(size() + 1);
    for (auto lambda : rpo()) {
        for (auto succ : lambda->succs()) {
            if (succ->find_scope(this))
                link(lambda, succ);
        }
    }

    // link meta lambda to real entries
    if (entries.size() != 1) {
        for (auto e : entries)
            link(entry(), e);
    }
}

void Scope::build_rev_rpo(Array<Lambda*> entries) {
    // find exits
    std::vector<Lambda*> exits;
    auto cmp = [] (Lambda* l1, Lambda* l2) { return l1->gid() < l2->gid(); };
    std::sort(entries.begin(), entries.end(), cmp);
    auto is_entry = [&] (Lambda* lambda) { return std::binary_search(entries.begin(), entries.end(), lambda, cmp); };

    size_t i = 0, found = 0;
    for (auto lambda : rpo()) {
        for (auto succ : succs(lambda)) {
            if (!is_entry(succ))
                goto next;
        }
        exits.push_back(lambda);
        found = i;
next:
        ++i;
    }

    if (exits.size() != 1) {
        auto exit = world().meta_lambda();
        set_candidate(exit);
        exit->register_scope(this)->rpo_id = size();
        rpo_.push_back(exit);
        for (auto e : exits)                            // link meta lambda to exits
            link(e, exit);
    }

    rev_rpo_.resize(size());
    std::reverse_copy(rpo_.begin(), rpo_.end(), rev_rpo_.begin());

    size_t n = size();
    if (exits.size() == 1) {
        std::swap(rev_rpo_.front(), rev_rpo_[found]);   // put exit to front
        n = rev_number(exits.front(), n);
    } else {
        for (auto exit : exits) {
            if (rev_rpo_id(exit) == size_t(-1))         // if not visited
                n = rev_number(exit, n);
        }
        n = rev_number(exit(), n);
    }
    assert(n == 0);

    // sort in reverse post-order
    std::sort(rev_rpo_.begin(), rev_rpo_.end(), [&] (Lambda* l1, Lambda* l2) { return rev_rpo_id(l1) < rev_rpo_id(l2); });
}

size_t Scope::rev_number(Lambda* cur, size_t i) {
    auto info = cur->find_scope(this);
    info->rev_rpo_id = size_t(-2);                      // mark as visisted
    for (auto pred : preds(cur)) {
        if (rev_rpo_id(pred) == size_t(-1))             // if not visited
            i = rev_number(pred, i);
    }

    assert(cur->scopes_.front().scope == this && "front item does not point to this scope");
    return info->rev_rpo_id = i-1;
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

const DomTree* Scope::domtree() const { return lazy(domtree_); }
const PostDomTree* Scope::postdomtree() const { return lazy(postdomtree_); }
const LoopTree* Scope::looptree() const { return lazy(looptree_); }

template<bool elide_empty>
void Scope::for_each(World& world, std::function<void(const Scope&)> f) {
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

template void Scope::for_each<true> (World&, std::function<void(const Scope&)>);
template void Scope::for_each<false>(World&, std::function<void(const Scope&)>);

}
