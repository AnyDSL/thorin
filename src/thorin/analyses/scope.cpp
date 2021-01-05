#include "thorin/analyses/scope.h"

#include <algorithm>

#include "thorin/rewrite.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

Scope::Scope(Def* entry)
    : world_(entry->world())
    , entry_(entry)
    , exit_(world().nom_lam(world().cn(world().bot_kind()), world_.dbg("exit")))
{
    run();
}

Scope::~Scope() {}

void Scope::run() {
    unique_queue<DefSet&> queue(bound_);
    queue.push(entry_->var());

    while (!queue.empty()) {
        for (auto use : queue.pop()->uses()) {
            if (use != entry_ && use != exit_) queue.push(use);
        }
    }
}

void Scope::calc_bound() const {
    if (has_bound_) return;
    has_bound_ = true;

    DefSet live;
    unique_queue<DefSet&> queue(live);

    auto enqueue = [&](const Def* def) {
        if (def->is_const()) return;

        if (bound_.contains(def))
            queue.push(def);
        else
            free_defs_.emplace(def);
    };

    for (auto op : entry()->extended_ops())
        enqueue(op);

    while (!queue.empty()) {
        for (auto op : queue.pop()->extended_ops())
            enqueue(op);
    }

    swap(live, bound_);
}

void Scope::calc_free() const {
    if (has_free_) return;
    has_free_ = true;

    unique_queue<DefSet> queue;

    auto enqueue = [&](const Def* def) {
        if (def->is_const()) return;

        if (auto var = def->isa<Var>())
            free_vars_.emplace(var);
        else if (auto nom = def->isa_nom())
            free_noms_.emplace(nom);
        else
            queue.push(def);
    };

    for (auto free : free_defs())
        enqueue(free);

    while (!queue.empty()) {
        for (auto op : queue.pop()->extended_ops())
            enqueue(op);
    }
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

Stream& Scope::stream(Stream& s) const { return schedule(*this).stream(s); }

template void Streamable<Scope>::dump() const;
template void Streamable<Scope>::write() const;

bool is_free(const Var* var, const Def* def) {
    // optimize common cases
    if (def == var) return true;
    for (auto p : var->nom()->vars())
        if (p == var) return true;

    Scope scope(var->nom());
    return scope.bound(def);
}

}
