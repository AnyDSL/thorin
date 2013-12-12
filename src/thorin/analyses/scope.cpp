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
            if (lambda != scope->entries().front() && scope->contains(lambda)) {
                top_level.erase(lambda);
                goto next_lambda;
            }
    next_lambda:;
    }

    Array<Lambda*> result(top_level.size());
    std::copy(top_level.begin(), top_level.end(), result.begin());
    return result;
}

Scope::Scope(Lambda* entry)
    : world_(entry->world())
    , num_entries_(1)
    , num_exits_(-1)
{
    identify_scope({entry});
    rpo_numbering({entry});
}

Scope::Scope(World& world, ArrayRef<Lambda*> entries)
    : world_(world)
    , num_entries_(entries.size())
{
    identify_scope(entries);
    rpo_numbering(entries);
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    LambdaSet lambdas;
    for (auto entry : entries)
        lambdas.insert(entry);
    for (auto entry : entries)
        collect(entry);
#ifndef NDEBUG
    for (auto lambda : rpo())
        assert(contains(lambda));
#endif
}

void Scope::collect(Lambda* entry) {
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

void Scope::rpo_numbering(ArrayRef<Lambda*> entries) {
    LambdaSet lambdas;

    for (auto entry : entries)
        lambdas.visit(entry);

    size_t num = 0;
    for (auto entry : entries)
        num = po_visit<true>(lambdas, entry, num);

    for (size_t i = entries.size(); i-- != 0;)
        sid_[entries[i]].sid = num++;

    assert(num <= size());
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
    
    // sort rpo_ according to sid_ which now holds the rpo number
    std::sort(rpo_.begin(), rpo_.end(), [&](Lambda* l1, Lambda* l2) { return sid_[l1].sid < sid_[l2].sid; });

    // discard unreachable lambdas
    rpo_.resize(num);

#ifndef NDEBUG
    std::set<size_t> done;
    for (auto p : sid_) {
        if (p.second.sid != size_t(-1))
            assert(done.insert(p.second.sid).second);
    }
#endif
}

template<bool forwards>
size_t Scope::po_visit(LambdaSet& set, Lambda* cur, size_t i) const {
    for (auto succ : forwards ? cur->succs() : cur->preds()) {
        if (in_scope_.contains(succ) && !set.contains(succ))
            i = number<forwards>(set, succ, i);
    }
    return i;
}

template<bool forwards>
size_t Scope::number(LambdaSet& set, Lambda* cur, size_t i) const {
    set.visit(cur);
    i = po_visit<forwards>(set, cur, i);
    return forwards ? (sid_[cur].sid = i) + 1 : (sid_[cur].backwards_sid = i) - 1;
}

#define THORIN_SCOPE_SUCC_PRED(succ) \
ArrayRef<Lambda*> Scope::succ##s(Lambda* lambda) const { \
    assert(contains(lambda));  \
    if (succ##s_.data() == nullptr) { \
        succ##s_.alloc(size()); \
        for (auto lambda : rpo_) { \
            Lambdas all_succ##s = lambda->succ##s(); \
            auto& succ##s = succ##s_[sid(lambda)]; \
            succ##s.alloc(all_succ##s.size()); \
            size_t i = 0; \
            for (auto succ : all_succ##s) { \
                if (contains(succ)) \
                    succ##s[i++] = succ; \
            } \
            succ##s.shrink(i); \
        } \
    } \
    return succ##s_[sid(lambda)];  \
}

THORIN_SCOPE_SUCC_PRED(succ)
THORIN_SCOPE_SUCC_PRED(pred)

ArrayRef<Lambda*> Scope::backwards_rpo() const {
    if (!backwards_rpo_) {
        backwards_rpo_ = new Array<Lambda*>(size());

        std::vector<Lambda*> exits;

        for (auto lambda : rpo()) {
            if (num_succs(lambda) == 0)
                exits.push_back(lambda);
        }

        num_exits_ = exits.size();

        // number all lambdas in postorder
        LambdaSet lambdas;

        size_t num = 0;
        for (auto exit : exits) {
            lambdas.visit(exit);
            sid_[exit].backwards_sid = num++;
        }

        num = size() - 1;
        for (auto exit : exits)
            num = po_visit<false>(lambdas, exit, num);

        assert(num + 1 == num_exits());

        std::copy(rpo_.begin(), rpo_.end(), backwards_rpo_->begin());
        std::sort(backwards_rpo_->begin(), backwards_rpo_->end(), [&](Lambda* l1, Lambda* l2) {
            return sid_[l1].backwards_sid < sid_[l2].backwards_sid;
        });
    }

    return *backwards_rpo_;
}

//------------------------------------------------------------------------------

bool Scope::is_entry(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].sid < num_entries(); }
bool Scope::is_exit(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].backwards_sid < num_exits(); }

size_t Scope::sid(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].sid; }
size_t Scope::backwards_sid(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].backwards_sid; }

//------------------------------------------------------------------------------

} // namespace thorin
