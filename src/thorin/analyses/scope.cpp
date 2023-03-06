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

Scope::Scope(Continuation* entry) : Scope(entry, std::make_shared<ScopesForest>()) {
    run();
}

Scope::Scope(Continuation* entry, std::shared_ptr<ScopesForest> forest)
    : world_(entry->world())
    , forest_(forest)
    , entry_(entry)
{}

Scope::~Scope() {}

Scope& Scope::update() {
    defs_.clear();
    free_frontier_.clear();
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

ParamSet Scope::search_free_variables_nonrec(bool root) const {
    bool valid_results = true;

    //world().WLOG("free variables analysis: searching transitive ops of: {}", entry());
    ParamSet free_params;
    forest_->stack_.push_back(entry());

    unique_queue<DefSet> queue;

    for (auto def : free_frontier_)
        queue.push(def);

    if (free_params_) {
        forest_->stack_.pop_back();
        return *free_params_;
    }

    while (!queue.empty()) {
        auto free_def = queue.pop();
        assert(!contains(free_def));

        if (free_def == entry())
            continue;

        if (auto param = free_def->isa<Param>(); param && !param->continuation()->dead_)
            free_params.emplace(param);
        else if (auto cont = free_def->isa_nom<Continuation>()) {
            // the free variables analysis can be recursive, but it's not necessary to inspect our own scope again ...
            if (cont == entry())
                continue;

            // if we hit the recursion wall, the results for this free variable search are only valid for the callee
            if (std::find(forest_->stack_.begin(), forest_->stack_.end(), cont) != forest_->stack_.end()) {
                assert(!root);
                valid_results = false;
                continue;
            }
            Scope& scope = forest_->get_scope(cont, forest_);
            assert(scope.defs().size() > 0 || !scope.entry()->has_body());

            // When we have a free continuation in the body of our fn, their free variables are also free in us
            ParamSet fp = scope.search_free_variables_nonrec(false);
            valid_results &= scope.free_params_.get() != nullptr;
            for (auto fv: fp) {
                // well except if these are our own ;)
                if (fv->continuation() == entry())
                    continue;
                // (those variables have to be free here! otherwise that continuation should be in this scope and not free)
                if (contains(fv))
                    world().WLOG("Potentially broken scoping: free variable {} showed up in the free variables of {} despite that continuation being part of its scope", fv, entry());
                free_params.insert(fv);
            }
        }
        else {
            for (auto op : free_def->ops()) {
                if (op == entry())
                    continue;
                assert(!contains(op));
                queue.push(op);
            }
        }
    }

    //world().WLOG("free variables analysis: done with : {} (hit_wall={})", entry(), hit_recursion_wall);

    assert(forest_->stack_.back() == entry());
    forest_->stack_.pop_back();

    if (valid_results) {
        free_params_ = std::make_unique<ParamSet>(free_params);
        return free_params;
    }

    for (auto fp : free_params) {
        assert(fp->continuation() != entry());
    }

    return free_params;
}

const ParamSet& Scope::free_params() const {
    if (!free_params_) {
        free_params_ = std::make_unique<ParamSet>(search_free_variables_nonrec(true));
    }

    return *free_params_;
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(Scope&)> f) {
    auto forest = std::make_shared<ScopesForest>();
    for (auto cont : world.copy_continuations()) {
        if (elide_empty && !cont->has_body())
            continue;
        assert(forest->stack_.empty());
        //forest->scopes_.clear();
        auto& scope = forest->get_scope(cont, forest);
        //Scope scope(cont);
        assert(forest->stack_.empty());
        if(!scope.has_free_params()) {
            assert(forest->stack_.empty());
            f(scope);
        }
    }
}

template void Scope::for_each<true> (const World&, std::function<void(Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(Scope&)>);

Scope& ScopesForest::get_scope(Continuation* entry, std::shared_ptr<ScopesForest>& self) {
    if (scopes_.contains(entry)) {
        auto existing = scopes_.find(entry);
        assert((size_t)(existing->second.get()) != 0xbebebebe00000000);
        return *existing->second;
    }
    auto scope = std::make_unique<Scope>(entry, self);
    Scope* ptr = scope.get();
    ptr->run();
    scopes_[entry] = std::move(scope);
    return *ptr;
}

}
