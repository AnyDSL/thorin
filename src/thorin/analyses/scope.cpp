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

Scope::Scope(Lambda* entry, bool forwards)
    : world_(entry->world())
    , forwards_(forwards)
{
    identify_scope(entry);
    build_cfg();
    uce();
    find_exits();
    rpo_numbering();
}

void Scope::identify_scope(Lambda* entry) {
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

#ifndef NDEBUG
    for (auto lambda : rpo())
        assert(contains(lambda));
#endif
}

void Scope::build_cfg() {
    for (auto lambda : rpo_) {
        Lambdas all_succs = lambda->succs();
        for (auto succ : all_succs) {
            if (contains(succ))
                link(lambda, succ);
        }
    }
}

void Scope::uce() {
    LambdaSet reachable;
    std::queue<Lambda*> queue;
    queue.push(entry());
    reachable.insert(entry());

    while (!queue.empty()) {
        Lambda* lambda = queue.front();
        queue.pop();

        std::cout << lambda->unique_name() << std::endl;
        for (auto succ : succs(lambda)) {
            std::cout << "succ: " << succ->unique_name() << std::endl;
            if (!reachable.contains(succ)) {
                queue.push(succ);
                reachable.insert(succ);
            }
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

    auto exit  = world().meta_lambda({}, "exit");
    rpo_.push_back(exit);
    in_scope_.insert(exit);

    for (auto e : exits) {
        e->ignore(exit);
        link(e, exit);
    }
}

void Scope::rpo_numbering() {
    LambdaSet lambdas;
    lambdas.insert(entry());
    int num = rpo_.size();
    num = po_visit(lambdas, entry(), num);

    assert(num == size());
    assert(num == lambdas.size());

    // sort rpo_ according to sid which now holds the rpo number
    std::sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) { return sid(l1) < sid(l2); });
}

int Scope::po_visit(LambdaSet& set, Lambda* cur, int i) {
    for (auto succ : succs(cur)) {
        if (!set.contains(succ)) {
            set.insert(cur);
            i = po_visit(set, cur, i);
            (*(forwards() ? &sid_ : &reverse_sid_))[cur] = i;
        }
    }
    return i-1;
}

//------------------------------------------------------------------------------

Array<Lambda*> top_level_lambdas(World& world) {
    // trivial implementation, but works
    // TODO: nicer version with Fibonacci heaps
    AutoVector<Scope*> scopes;
    for (auto lambda : world.lambdas()) {
        std::cout << lambda->unique_name() << std::endl;
        scopes.push_back(new Scope(lambda));
    }

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

Lambda* top_lambda(World& world) { return world.meta_lambda(top_level_lambdas(world), "top"); }

} // namespace thorin
