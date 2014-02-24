#include "thorin/analyses/scope.h"

#include <algorithm>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

Scope::Scope(World& world, ArrayRef<Lambda*> entries, int mode)
    : world_(world)
    , mode_(mode)
{
    identify_scope(entries);
    build_succs();

    auto entry = world.meta_lambda();
    rpo_.push_back(entry);
    in_scope_.insert(entry);

    auto& pair = succs_[entry];
    for (auto e : entries)
        pair.second.push_back(e);
    pair.first = pair.second.size();
    assert(succs_[entry].second.size() == entries.size());

    uce(entry);
    auto exit = find_exits();
    build_preds();
    rpo_numbering(entry, exit);
}

Scope::Scope(Lambda* entry, int mode)
    : world_(entry->world())
    , mode_(mode)
{
    identify_scope({entry});
    build_succs();
    uce(entry);
    auto exit = find_exits();
    build_preds();
    rpo_numbering(entry, exit);
}

Scope::~Scope() {
    std::vector<Lambda*> remove;
    if (!entry()->empty() && entry()->to()->isa<Bottom>())
        world().destroy(entry());
    if (has_unique_exit() && exit()->to()->isa<Bottom>())
        world().destroy(exit());
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
    for (auto lambda : rpo()) assert(contains(lambda));
#endif
}

void Scope::build_succs() {
    for (auto lambda : rpo_) {
        auto& pair = succs_[lambda];
        for (auto succ : lambda->direct_succs()) {
            if (contains(succ))
                pair.second.push_back(succ);
        }
        pair.first = pair.second.size();
        for (auto succ : lambda->indirect_succs()) {
            if (contains(succ))
                pair.second.push_back(succ);
        }
    }
}

void Scope::build_preds() {
    assert(is_forward() && "TODO");
    for (auto lambda : rpo_) {
        for (auto succ : direct_succs(lambda))
            preds_[succ].second.push_back(lambda);
    }
    for (auto lambda : rpo_) {
        auto& pair = preds_[lambda];
        pair.first = pair.second.size();
    }
    for (auto lambda : rpo_) {
        for (auto succ : indirect_succs(lambda))
            preds_[succ].second.push_back(lambda);
    }
}

void Scope::uce(Lambda* entry) {
    LambdaSet reachable;

    {   // mark all reachable stuff
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
    }
    {   // transitively erase all non-reachable stuff
        std::queue<Def> queue;
        auto insert = [&] (Def def) { 
            queue.push(def); 
            if (Lambda* lambda = def->isa_lambda())
                for (auto param : lambda->params())
                    queue.push(param);
        };

        for (auto lambda : rpo_) {
            if (!reachable.contains(lambda))
                insert(lambda);
        }

        while (!queue.empty()) {
            Def def = queue.front();
            queue.pop();
            int num = in_scope_.erase(def);
            assert(num = 1);

            for (auto use : def->uses()) {
                if (contains(use))
                    insert(use);
            }
        }
    }

    rpo_.resize(reachable.size(), nullptr);
    std::copy(reachable.begin(), reachable.end(), rpo_.begin());
#ifndef NDEBUG
    for (auto lambda : rpo()) assert(contains(lambda));
#endif
}

Lambda* Scope::find_exits() {
    if (!has_unique_exit())
        return nullptr;

    assert(false && "TODO");
    LambdaSet exits;

    for (auto lambda : rpo()) {
        if (num_succs(lambda) == 0)
            exits.insert(lambda);
    }

    auto exit  = world().meta_lambda();
    rpo_.push_back(exit);
    in_scope_.insert(exit);

    // TODO direct/indirect succs/preds
    //for (auto e : exits)
        //link_succ(e, exit);

    assert(!exits.empty() && "TODO");
    return exit;
}

void Scope::rpo_numbering(Lambda* entry, Lambda* exit) {
    LambdaSet visited;

    int num = size()-1;
    if (exit != nullptr) {
        visited.insert(exit);
        assign_sid(exit, num--);
    }
    num = po_visit(visited, entry, num);
    assert(size() == visited.size() && num == -1);

    // sort rpo_ according to sid which now holds the rpo number
    std::stable_sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) { return sid(l1) < sid(l2); });

#ifndef NDEBUG
    // double check sids
    for (int i = 0, e = size(); i != e; ++i)
        assert(sid(rpo_[i]) == i);
#endif
}

int Scope::po_visit(LambdaSet& visited, Lambda* cur, int i) {
    assert(!visited.contains(cur));
    visited.insert(cur);

    for (auto succ : succs(cur)) {
        if (!visited.contains(succ))
            i = po_visit(visited, succ, i);
    }

    assign_sid(cur, i);
    return i-1;
}

} // namespace thorin
