#include "thorin/analyses/scope.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

//------------------------------------------------------------------------------

Scope::Scope(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
{
#ifndef NDEBUG
    for (auto entry : entries) assert(!entry->is_proxy());
#endif

    identify_scope(entries);
    build_succs();

    Lambda* entry;
    if (entries.size() == 1)
        entry = entries.front();
    else {
        entry = world.meta_lambda();
        rpo_.push_back(entry);
        in_scope_.insert(entry);

        for (auto e : entries)
            link_succ(entry, e);
    }

    uce(entry);
    build_preds();
    auto exit = find_exit();
    link_exit(entry, exit);
    ScopeView(*this,  true).rpo_numbering(entry);
    ScopeView(*this, false).rpo_numbering(exit);
}

Scope::~Scope() {
    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        world().destroy(entry());
    if (exit() != entry() && !exit()->empty() && exit()->to()->isa<Bottom>())
        world().destroy(exit());
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    for (auto entry : entries) {
        if (!in_scope_.contains(entry)) {
            std::queue<Def> queue;

            auto insert_lambda = [&] (Lambda* lambda) {
                assert(!lambda->is_proxy());
                for (auto param : lambda->params()) {
                    if (!param->is_proxy()) {
                        in_scope_.insert(param);
                        queue.push(param);
                    }
                }

                assert(std::find(rpo_.begin(), rpo_.end(), lambda) == rpo_.end());
                rpo_.push_back(lambda);
            };

            insert_lambda(entry);
            in_scope_.insert(entry);

            while (!queue.empty()) {
                auto def = queue.front();
                queue.pop();
                for (auto use : def->uses()) {
                    if (!in_scope_.contains(use)) {
                        if (auto ulambda = use->isa_lambda())
                            insert_lambda(ulambda);
                        in_scope_.insert(use);
                        queue.push(use);
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    for (auto lambda : rpo_) assert(contains(lambda));
#endif
}

void Scope::build_succs() {
    for (auto lambda : rpo_) {
        for (auto succ : lambda->succs()) {
            if (contains(succ))
                link_succ(lambda, succ);
        }
    }
}

void Scope::build_preds() {
    for (auto lambda : rpo_) {
        for (auto succ : succs_[lambda])
            link_pred(lambda, succ);
    }
}

void Scope::uce(Lambda* entry) {
    auto reachable = ScopeView(*this, true).reachable(entry);
    // transitively mark all reachable stuff

    DefSet new_in_scope;
    std::queue<Def> queue;
    auto enqueue = [&] (Def def) { 
        assert(in_scope_.contains(def) && !new_in_scope.contains(def));
        queue.push(def);
        new_in_scope.insert(def);
    };

    for (auto lambda : reachable) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                enqueue(param);
        }

        enqueue(lambda);
    }

    while (!queue.empty()) {
        Def def = queue.front();
        queue.pop();
        for (auto op : def->ops()) {
            if (in_scope_.contains(op) && !new_in_scope.contains(op))
                enqueue(op);
        }
    }

    rpo_.resize(reachable.size(), nullptr);
    reverse_rpo_.resize(reachable.size(), nullptr);
    std::copy(reachable.begin(), reachable.end(), rpo_.begin());
    std::copy(reachable.begin(), reachable.end(), reverse_rpo_.begin());
    swap(new_in_scope, in_scope_);

#ifndef NDEBUG
    for (auto lambda : rpo_) 
        assert(contains(lambda));
#endif
}

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
        reverse_rpo_.push_back(exit);
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

//------------------------------------------------------------------------------

LambdaSet ScopeView::reachable(Lambda* entry) {
    LambdaSet set;
    std::queue<Lambda*> queue;
    auto insert = [&] (Lambda* lambda) { queue.push(lambda); set.insert(lambda); };
    insert(entry);

    while (!queue.empty()) {
        Lambda* lambda = queue.front();
        queue.pop();

        for (auto succ : succs(lambda)) {
            if (!set.contains(succ))
                insert(succ);
        }
    }
    return set;
}

void ScopeView::rpo_numbering(Lambda* entry) {
    LambdaSet visited;
    visit_first(visited, entry);
    int num = po_visit(visited, entry, size()-1);
    assert(size() == visited.size() && num == -1);

    // sort rpo according to sid which now holds the rpo number
    std::stable_sort(rpo_vector().begin(), rpo_vector().end(), [&](Lambda* l1, Lambda* l2) { return sid()[l1] < sid()[l2]; });

#ifndef NDEBUG
    for (int i = 0, e = size(); i != e; ++i)
        assert(sid()[rpo()[i]] == i && "double check of sids went wrong");
#endif
}

int ScopeView::po_visit(LambdaSet& done, Lambda* cur, int i) {
    for (auto succ : succs(cur)) {
        if (!visit(done,  succ))
            i = po_visit(done, succ, i);
    }

    sid()[cur] = i;
    return i-1;
}

//------------------------------------------------------------------------------

}
