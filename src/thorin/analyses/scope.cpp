#include "thorin/analyses/scope.h"

#include <algorithm>
#include <iostream>

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

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

Scope::Scope(World& world) 
    : world_(world)
    , num_entries_(0)
{
    top_level_ = world.lambdas();

    for (auto lambda : world.lambdas())
        if (!set_.contains(lambda))
            collect(lambda);

    std::vector<Lambda*> entries;
    std::copy(top_level_.begin(), top_level_.end(), std::inserter(entries, entries.begin()));
    num_entries_ = entries.size();
    rpo_numbering(entries);
}

void Scope::identify_scope(ArrayRef<Lambda*> entries) {
    for (auto entry : entries)
        collect(entry);
}

void Scope::collect(Lambda* lambda) {
    if (set_.contains(lambda))
        return;

    set_.insert(lambda);
    rpo_.push_back(lambda);

    std::queue<Def> queue;
    for (auto param : lambda->params()) {
        if (!param->is_proxy()) {
            set_.insert(param);
            queue.push(param);
        }
    }

    LambdaSet lambdas;
    while (!queue.empty()) {
        auto def = queue.front();
        queue.pop();
        for (auto use : def->uses()) {
            if (auto ulambda = use->isa_lambda()) {
                if (ulambda != lambda)
                    lambdas.insert(ulambda);
            } else if (!set_.contains(use)) {
                set_.insert(use);
                queue.push(use);
            }
        }
    }

    for (auto lambda : lambdas) {
        top_level_.erase(lambda);
        collect(lambda);

        for (auto pred : lambda->preds()) {
            if (!set_.contains(pred)) {
                for (auto op : pred->ops()) {
                    if (set_.contains(op)) {
                        top_level_.erase(pred);
                        collect(pred);
                        goto next_pred;
                    }
                }
            }
next_pred:;
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
    assert(num >= 0);

    // convert postorder number to reverse postorder number
    for (auto lambda : rpo()) {
        if (lambdas.contains(lambda)) {
            sid_[lambda].sid = num - 1 - sid_[lambda].sid;
        } else { // lambda is unreachable
            set_.erase(lambda);
            sid_.erase(lambda);
        }
    }
    
    // sort rpo_ according to sid_ which now holds the rpo number
    std::sort(rpo_.begin(), rpo_.end(), [&](const Lambda* l1, const Lambda* l2) { return sid_[l1].sid < sid_[l2].sid; });

    // discard unreachable lambdas
    rpo_.resize(num);
}

template<bool forwards>
size_t Scope::po_visit(LambdaSet& set, Lambda* cur, size_t i) const {
    for (auto succ : forwards ? cur->succs() : cur->preds()) {
        if (contains(succ) && !set.contains(succ))
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
        std::sort(backwards_rpo_->begin(), backwards_rpo_->end(), [&](const Lambda* l1, const Lambda* l2) {
            return sid_[l1].backwards_sid < sid_[l2].backwards_sid;
        });
    }

    return *backwards_rpo_;
}

size_t Scope::mark() const {
    std::queue<Def> queue;
    const auto pass = world().new_pass();

    for (auto lambda : rpo()) {
        lambda->visit_first(pass);
        queue.push(lambda);

        for (auto param : lambda->params()) {
            param->visit_first(pass);
            queue.push(param);
        }

        mark_down(pass, queue);
    }

    return pass;
}

//------------------------------------------------------------------------------

LambdaSet Scope::resolve_lambdas() const
{
    LambdaSet result;
    std::copy(rpo_.begin(), rpo_.end(), std::inserter(result, result.begin()));
    return result;
}

bool Scope::is_entry(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].sid < num_entries(); }
bool Scope::is_exit(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].backwards_sid < num_exits(); }

size_t Scope::sid(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].sid; }
size_t Scope::backwards_sid(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda].backwards_sid; }

const DomTree& Scope::domtree() const { return domtree_ ? *domtree_ : *(domtree_ = new DomTree(*this)); }
const PostDomTree& Scope::postdomtree() const { return postdomtree_ ? *postdomtree_ : *(postdomtree_ = new PostDomTree(*this)); }
const LoopTree& Scope::looptree() const { return looptree_ ? *looptree_ : *(looptree_ = new LoopTree(*this)); }

//------------------------------------------------------------------------------

} // namespace thorin
