#include "thorin/analyses/scope.h"

#include <algorithm>
#include <fstream>

#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

Scope::Scope(Continuation* entry) : world_(entry->world()), root(std::make_unique<ScopesForest>(world())), forest_(*root), entry_(entry) {
    run();
}

Scope::Scope(Continuation* entry, ScopesForest& forest)
    : world_(entry->world())
    , forest_(forest)
    , entry_(entry)
{}

Scope::~Scope() {}

Scope& Scope::update() {
    defs_.clear();
    free_frontier_.clear();
    first_free_param_ = nullptr;
    free_params_ = nullptr;
    cfa_         = nullptr;
    run();
    return *this;
}

DefSet Scope::potentially_contained() const {
    DefSet potential_defs;
    std::queue<const Def*> queue;

    auto enqueue = [&] (const Def* def) {
        if (potential_defs.insert(def).second) {
            queue.push(def);
        }
    };

    for (auto param : entry()->params())
        enqueue(param);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (def != entry()) {
            for (auto use: def->uses())
                enqueue(use);
        }
    }

    return potential_defs;
}

void Scope::run() {
    DefSet potential_defs = potentially_contained();

    unique_queue<DefSet> queue;

    if (entry()->has_body())
        queue.push(entry()->body());
    queue.push(entry()->filter());

    defs_.insert(entry());

    while (!queue.empty()) {
        auto def = queue.pop();

        if (potential_defs.contains(def)) {
            defs_.insert(def);
            for (auto op : def->ops())
                queue.push(op);
        } else {
            free_frontier_.insert(def);
        }
    }
}

void Scope::verify() {
    for (auto def : defs()) {
        if (auto cont = def->isa_nom<Continuation>()) {
            if (cont == entry())
                continue;
            assert(!forest_.get_scope(cont).contains(entry()));
        }
    }
}

static auto first_or_null = [](ParamSet& set) -> const Param* {
    for (auto p : set) {
        return p;
    }
    return nullptr;
};

// searches for free variable in this scope, starting at the free frontier and transitively searching the scopes of the free continuations we encounter
// stop_after_first is used when we only care about whether this is a top-level scope (ie has no free params) or not, we can stop as soon as one param is found
template<bool stop_after_first>
std::tuple<ParamSet, bool> Scope::search_free_params() const {
    if (free_params_) {
        //world().WLOG("free variables analysis: reusing cached results for {}", entry());
        return std::make_tuple(*free_params_, true);
    }

    if (stop_after_first && first_free_param_) {
        ParamSet one_or_zero_params;
        if (*first_free_param_ != nullptr)
            one_or_zero_params.insert(first_free_param());
        return std::make_tuple(one_or_zero_params, true);
    }

    ParamSet free_params;
    /// as much as possible, we'd like to keep the results of those searches, but in the recursive case it's not always possible:
    /// if there is a cycle, we stop when we encounter a continuation we're already searching the free variables for
    /// this variable keeps track of whether we did that or not, if we didn't and this is a full search, we can safely save the results
    bool thorough = true;
    bool is_root = forest_.stack_.empty();
    //world().WLOG("free variables analysis: searching transitive ops of: {} (root={}, depth={})", entry(), root, forest_->stack_.size());
    forest_.stack_.push_back(entry());

    unique_queue<DefSet> queue;

    for (auto def : free_frontier_)
        queue.push(def);

    while (!queue.empty()) {
        auto free_def = queue.pop();
        assert(!contains(free_def));

        if (auto param = free_def->isa<Param>(); param && !param->continuation()->dead_) {
            free_params.insert(param);
            if (stop_after_first)
                break;
        } else if (auto cont = free_def->isa_nom<Continuation>()) {
            // the free variables analysis can be recursive, but it's not necessary to inspect our own scope again ...
            assert(cont != entry());

            // if we hit the recursion wall, the results for this free variable search are only valid for the callee
            if (std::find(forest_.stack_.begin(), forest_.stack_.end(), cont) != forest_.stack_.end()) {
                //world().WLOG("free variables analysis: skipping {} to prevent infinite recursion", cont);
                thorough = false;
                continue;
            }

            Scope& scope = forest_.get_scope(cont);
            assert(!scope.defs().empty() || !scope.entry()->has_body());

            // When we have a free continuation in the body of our fn, their free variables are also free in us
            auto [callee_free_params, callee_results_thorough] = scope.search_free_params<stop_after_first>();
            if (!is_root)
                thorough &= callee_results_thorough;

            for (auto p: callee_free_params) {
                // (those variables have to be free here! otherwise that continuation should be in this scope and not free)
                if (contains(p)) {
                    world().WLOG("Potentially broken scoping: free variable {} showed up in the free variables of {} despite that continuation being part of its scope", p, entry());
                    assert(false);
                }
                free_params.insert(p);
                if (stop_after_first)
                    break;
            }
        }
        else {
            for (auto op : free_def->ops()) {
                // the entry might be referenced by the outside world, but that's completely fine
                if (op == entry())
                    continue;
                assert(!contains(op));
                queue.push(op);
            }
        }
    }

    //world().WLOG("free variables analysis: done with : {} (hit_wall={})", entry(), hit_recursion_wall);

    assert(forest_.stack_.back() == entry());
    forest_.stack_.pop_back();

    // save the results if we can
    if (thorough) {
        if (stop_after_first) {
            first_free_param_ = std::make_optional<const Param*>(first_or_null(free_params));
            return std::make_tuple(free_params, true);
        } else {
            free_params_ = std::make_unique<ParamSet>(std::move(free_params));
            return std::make_tuple(*free_params_, true);
        }
    }

    for (auto fp : free_params) {
        assert(fp->continuation() != entry());
    }

    return std::make_tuple(free_params, thorough);
}

const ParamSet& Scope::free_params() const {
    if (!free_params_) {
        auto [set, valid] = search_free_params<false>();
        assert(valid);
        free_params_ = std::make_unique<ParamSet>(set);
    }

    return *free_params_;
}

const Param* Scope::first_free_param() const {
    if (!first_free_param_) {
        // if we already computed the full free params list, let's reuse that !
        if (free_params_) {
            first_free_param_ = std::make_optional<const Param*>(first_or_null(*free_params_));
        } else {
            auto [set, valid] = search_free_params<true>();
            assert(valid);
            first_free_param_ = std::make_optional<const Param*>(first_or_null(set));
        }
    }

    return *first_free_param_;
}

bool Scope::has_free_params() const {
    return first_free_param() != nullptr;
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

Continuation* Scope::parent_scope() const {
    if (!parent_scope_) {
        ContinuationSet candidates_set;
        std::queue<Continuation*> candidates;

        for (auto param : free_params()) {
            if(candidates_set.insert(param->continuation()).second)
                candidates.push(param->continuation());
        }

        if (candidates.empty())
            parent_scope_ = std::make_optional<Continuation*>();
        else {
            while (true) {
                auto candidate = pop(candidates);
                // when there is only one candidate left, that's our parent
                if (candidates.empty()) {
                    parent_scope_ = std::make_optional<Continuation*>(candidate);
                    break;
                }

                auto other_candidate = pop(candidates);
                assert(candidate != other_candidate);
                if (forest_.get_scope(candidate).contains(other_candidate))
                    candidates.push(other_candidate);
                else {
                    assert(forest_.get_scope(other_candidate).contains(candidate) && "a scope cannot be nested in two unrelated parent scopes");
                    candidates.push(candidate);
                }
            }
        }
    }

    return *parent_scope_;
}

template<bool elide_empty>
void ScopesForest::for_each(std::function<void(Scope&)> f) {
    for (auto cont : world_.copy_continuations()) {
        if (elide_empty && !cont->has_body())
            continue;
        assert(stack_.empty());
        //forest->scopes_.clear();
        auto& scope = get_scope(cont);
        //Scope scope(cont);
        assert(stack_.empty());
        if(!scope.has_free_params()) {
            assert(stack_.empty());
            f(scope);
        }
    }
}

template void ScopesForest::for_each<true> ( std::function<void(Scope&)>);
template void ScopesForest::for_each<false>( std::function<void(Scope&)>);

Scope& ScopesForest::get_scope(Continuation* entry) {
    if (scopes_.contains(entry)) {
        auto existing = scopes_.find(entry);
        assert((size_t)(existing->second.get()) != 0xbebebebe00000000);
        return *existing->second;
    }
    auto scope = std::make_unique<Scope>(entry, *this);
    Scope* ptr = scope.get();
    ptr->run();
    scopes_[entry] = std::move(scope);
    if (stack_.empty())
       ptr->verify();
    return *ptr;
}

}
