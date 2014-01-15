#include "thorin/analyses/scope.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

Array<Lambda*> top_level_lambdas(World& world) {
    // trivial implementation, but works
    // TODO: nicer version with Fibonacci heaps
    AutoVector<Scope*> scopes;
    for (auto lambda : world.lambdas())
        scopes.push_back(new Scope(lambda));

    // check for top_level lambdas
    LambdaSet top_level = world.lambdas();
    for (auto lambda : world.lambdas()) {
        for (auto scope : scopes)
            if (lambda != scope->body().front() && scope->contains(lambda)) {
                top_level.erase(lambda);
                goto next_lambda;
            }
next_lambda:;
    }

    Array<Lambda*> result(top_level.size());
    std::copy(top_level.begin(), top_level.end(), result.begin());
    return result;
}

template<bool forwards>
ScopeBase<forwards>::ScopeBase(Lambda* entry)
    : world_(entry->world())
{
    identify_scope({entry});
    build_cfg({entry});
}

template<bool forwards>
ScopeBase<forwards>::ScopeBase(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
{
    identify_scope(entries);
    build_cfg(entries);
}

template<bool forwards>
void ScopeBase<forwards>::identify_scope(ArrayRef<Lambda*> entries) {
    for (auto entry : entries) {
        if (!in_scope_.contains(entry)) {
            std::queue<Def> queue;

            auto insert_lambda = [&] (Lambda* lambda) {
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
    for (auto lambda : rpo())
        assert(contains(lambda));
#endif
}

template<bool forwards>
void ScopeBase<forwards>::build_cfg(ArrayRef<Lambda*> entries) {
    // don't add to in_scope_; this is an implementation detail
    auto entry = world().meta_lambda();
    rpo_.push_back(entry);

    for (auto e : entries)
        link(entry, e);

    for (auto lambda : rpo_) {
        Lambdas all_succs = lambda->succs();
        for (auto succ : all_succs) {
            if (contains(succ))
                link(lambda, succ);
        }
    }

    uce(entry);
    //rpo_numbering(entry);
}

template<bool forwards>
void ScopeBase<forwards>::uce(Lambda* entry) {
    LambdaSet reachable;
    std::queue<Lambda*> queue;
    queue.push(entry);
    reachable.insert(entry);

    while (!queue.empty()) {
        Lambda* lambda = queue.front();
        queue.pop();

        for (auto succ : succs(lambda)) {
            if (!reachable.contains(succ)) {
                queue.push(succ);
                reachable.insert(succ);
            }
        }
    }

    rpo_.clear();
    std::copy(reachable.begin(), reachable.end(), std::inserter(rpo_, rpo_.begin()));
}

template<bool forwards>
void ScopeBase<forwards>::find_exits() {
    std::vector<Lambda*> exits;

    for (auto lambda : rpo()) {
        if (num_succs(lambda) == 0)
            exits.push_back(lambda);
    }

    LambdaSet exiting;
    std::queue<Lambda*> queue;

    while (!queue.empty()) {
        Lambda* lambda = queue.front();
        queue.pop();
    }

    auto exit  = world().meta_lambda();
    rpo_.push_back(exit);

    for (auto e : exits)
        link(e, exit);
}

template<bool forwards>
void ScopeBase<forwards>::rpo_numbering(Lambda* entry) {
    LambdaSet lambdas;
    lambdas.insert(entry);
    size_t num = 0;
    num = po_visit(lambdas, entry, num);

    assert(num == size());
    assert(num == lambdas.size());

    // convert postorder number to reverse postorder number
    for (auto lambda : rpo()) {
        if (lambdas.contains(lambda)) {
            sid_[lambda].sid = num - 1 - sid_[lambda].sid;
            assert(sid_[lambda].sid < 1000);
        } else { // lambda is unreachable
            sid_[lambda].sid = size_t(-1);
            for (auto param : lambda->params())
                if (!param->is_proxy())
                    in_scope_.erase(param);
            in_scope_.erase(lambda);
        }
    }
    
    // sort rpo_ according to sid which now holds the rpo number
    std::sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) { return sid(l1) < sid(l2); });
}

//template<bool forwards>
//int ScopeBase<forwards>::po_visit(LambdaSet& set, Lambda* cur, size_t i) const {
    //for (auto succ : forwards ? succs(cur) : preds(cur)) {
        //if (!set.contains(succ))
            //i = number<forwards>(set, succ, i);
    //}
    //return i;
//}

//template<bool forwards>
//size_t ScopeBase<forwards>::number(LambdaSet& set, Lambda* cur, size_t i) const {
    //set.visit(cur);
    //i = po_visit<forwards>(set, cur, i);
    //return forwards ? (sid_[cur].sid = i) + 1 : (sid_[cur].backwards_sid = i) - 1;
//}

} // namespace thorin
