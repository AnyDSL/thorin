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

Scope::Scope(Lambda* entry, bool unique_exit)
    : world_(entry->world())
    , sid_(sid_counter_++)
    , unique_exit_(unique_exit)
{
    assert(!entry->is_proxy());
    identify_scope(entry);
    number();
    build_cfg();
    if (unique_exit)
        build_backwards_rpo();
    build_in_scope();
    ++candidate_counter_;
}

Scope::~Scope() {
    for (auto lambda : rpo())
        lambda->unregister_scope(this);

    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        entry()->destroy_body();
    if (has_unique_exit() && exit() != entry() && !exit()->empty() && exit()->to()->isa<Bottom>())
        exit()->destroy_body();
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

#ifndef NDEBUG
    for (auto lambda : rpo())
        assert(is_candidate(lambda));
#endif
    assert(rpo().front() == entry);
}

void Scope::number() {
    size_t n = number(entry(), 0);

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

void Scope::build_cfg() {
    succs_.resize(size() + 1); // maybe we need one extra alloc for fake exit
    preds_.resize(size() + 1);
    for (auto lambda : rpo()) {
        for (auto succ : lambda->succs()) {
            if (succ->find_scope(this))
                link(lambda, succ);
        }
    }
}

void Scope::build_backwards_rpo() {
    // find exits
    std::vector<Lambda*> exits;

    size_t i = 0, found = 0;
    for (auto lambda : rpo()) {
        for (auto succ : succs(lambda)) {
            if (succ != entry())
                goto next;
        }
        exits.push_back(lambda);
        found = i;
next:
        ++i;
    }

    if (exits.empty())
        exits.push_back(rpo_.back());                   // HACK: simply choose the last one in rpo

    if (exits.size() != 1) {
        auto exit = world().meta_lambda();
        set_candidate(exit);
        exit->register_scope(this)->rpo_id = size();
        rpo_.push_back(exit);
        for (auto e : exits)                            // link meta lambda to exits
            link(e, exit);
    }

    backwards_rpo_.resize(size());
    std::reverse_copy(rpo_.begin(), rpo_.end(), backwards_rpo_.begin());

    size_t n = size();
    if (exits.size() == 1) {
        std::swap(backwards_rpo_.front(), backwards_rpo_[found]);   // put exit to front
        n = backwards_number(exits.front(), n);
    } else {
        for (auto exit : exits) {
            if (backwards_rpo_id(exit) == size_t(-1))         // if not visited
                n = backwards_number(exit, n);
        }
        n = backwards_number(exit(), n);
    }
    assert(n == 0);

    // sort in reverse post-order
    std::sort(backwards_rpo_.begin(), backwards_rpo_.end(), [&] (Lambda* l1, Lambda* l2) { 
        return backwards_rpo_id(l1) < backwards_rpo_id(l2); 
    });
}

size_t Scope::backwards_number(Lambda* cur, size_t i) {
    auto info = cur->find_scope(this);
    info->backwards_rpo_id = size_t(-2);                      // mark as visisted
    for (auto pred : preds(cur)) {
        if (backwards_rpo_id(pred) == size_t(-1))             // if not visited
            i = backwards_number(pred, i);
    }

    assert(cur->scopes_.front().scope == this && "front item does not point to this scope");
    return info->backwards_rpo_id = i-1;
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
