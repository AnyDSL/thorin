#include "thorin/analyses/scope.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

//------------------------------------------------------------------------------

Scope::Scope(World& world, ArrayRef<Lambda*> entries, bool forwards)
    : world_(world)
    , forwards_(forwards)
{
    identify_scope(entries);
    build_cfg(entries);

    auto entry = world.meta_lambda();
    rpo_.push_back(entry);
    in_scope_.insert(entry);

    for (auto e : entries)
        link(entry, e);

    uce(entry);
    find_exits();
    rpo_numbering(entry);
}

Scope::Scope(Lambda* entry, bool forwards)
    : world_(entry->world())
    , forwards_(forwards)
{
    identify_scope({entry});
    build_cfg({entry});
    uce(entry);
    find_exits();
    rpo_numbering(entry);
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    for (auto entry : entries) {
        if (!in_scope_.contains(entry)) {
            std::queue<Def> queue;

            auto insert = [&] (Lambda* lambda) {
                for (auto param : lambda->params()) {
                    if (!param->is_proxy()) {
                        in_scope_.insert(param);
                        queue.push(param);
                    }
                }

                assert(std::find(rpo_.begin(), rpo_.end(), lambda) == rpo_.end());
                rpo_.push_back(lambda);
            };

            insert(entry);
            in_scope_.insert(entry);

            while (!queue.empty()) {
                auto def = queue.front();
                queue.pop();
                for (auto use : def->uses()) {
                    if (!in_scope_.contains(use)) {
                        if (auto ulambda = use->isa_lambda())
                            insert(ulambda);
                        in_scope_.insert(use);
                        queue.push(use);
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    for (auto lambda : rpo())
        assert(contains(lambda));
#endif
}

void Scope::build_cfg(ArrayRef<Lambda*> entries) {
    for (auto lambda : rpo_) {
        auto succs = lambda->succs();
        for (auto succ : succs) {
            if (contains(succ))
                link(lambda, succ);
        }
    }
}

void Scope::uce(Lambda* entry) {
    LambdaSet reachable;
    std::queue<Lambda*> queue;

    auto insert = [&] (Lambda* lambda) { queue.push(lambda); reachable.insert(lambda); };
    insert(entry);

    while (!queue.empty()) {
        Lambda* lambda = queue.front();
        queue.pop();

        for (auto succ : succs(lambda)) {
            if (!reachable.contains(succ))
                insert(succ);
        }
    }

    rpo_.clear();
    std::copy(reachable.begin(), reachable.end(), std::inserter(rpo_, rpo_.begin()));
}

void Scope::find_exits() {
    LambdaSet exits;

    for (auto lambda : rpo()) {
        if (num_succs(lambda) == 0)
            exits.insert(lambda);
    }

    auto exit  = world().meta_lambda();
    rpo_.push_back(exit);
    in_scope_.insert(exit);

    for (auto e : exits)
        link(e, exit);

    assert(!exits.empty());
}

void Scope::rpo_numbering(Lambda* entry) {
    LambdaSet visited;

    visited.insert(entry);
    int num = rpo_.size();

    num = po_visit(visited, entry, num);
    assert(size() == visited.size());
    assert(num == 0);
    assign_sid(entry, num);

    // sort rpo_ according to sid which now holds the rpo number
    std::sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) { return sid(l1) < sid(l2); });

#ifndef NDEBUG
    // double check sids
    for (int i = 0, e = size(); i != e; ++i)
        assert(sid(rpo_[i]) == i);
#endif
}

int Scope::po_visit(LambdaSet& visited, Lambda* cur, int i) {
    for (auto succ : succs(cur)) {
        if (!visited.contains(succ)) {
            visited.insert(succ);
            i = po_visit(visited, succ, i);
            assign_sid(succ, i);
        }
    }
    return i-1;
}

//------------------------------------------------------------------------------

Array<Lambda*> top_level_lambdas(World& world) {
    // trivial implementation, but works
    // TODO: nicer version with Fibonacci heaps
    AutoVector<Scope*> scopes;
    std::vector<Lambda*> lambdas(world.lambdas().begin(), world.lambdas().end());
    for (auto lambda : lambdas)
        scopes.push_back(new Scope(lambda));

    // check for top_level lambdas
    LambdaSet top_level = world.lambdas();
    for (auto lambda : world.lambdas()) {
        for (auto scope : scopes)
            if (lambda != scope->entry() && scope->contains(lambda)) {
                top_level.erase(lambda);
                goto next_lambda;
            }
next_lambda:;
    }

    Array<Lambda*> result(top_level.size());
    std::copy(top_level.begin(), top_level.end(), result.begin());
    return result;
}

} // namespace thorin
