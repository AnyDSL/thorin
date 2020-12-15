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

Scope& Scope::update() {
    defs_.clear();
    free_defs_ = nullptr;
    free_      = nullptr;
    cfa_       = nullptr;
    run();
    return *this;
}

void Scope::run() {
    unique_queue<DefSet&> queue(defs_);
    queue.push(entry_->var());

    while (!queue.empty()) {
        for (auto use : queue.pop()->uses()) {
            if (use == entry_ || use == exit_) continue;
            queue.push(use);
        }
    }
}

const DefSet& Scope::free_defs() const {
    if (!free_defs_) {
        free_defs_ = std::make_unique<DefSet>();

        unique_queue<DefSet> queue;
        auto enqueue = [&](const Def* def) {
            if (def->is_const()) return;

            if (contains(def))
                queue.push(def);
            else
                free_defs_->emplace(def);
        };


        for (auto op : entry()->extended_ops())
            enqueue(op);

        while (!queue.empty()) {
            for (auto op : queue.pop()->extended_ops())
                enqueue(op);
        }
    }

    return *free_defs_;
}

const Scope::Free& Scope::free() const {
    if (!free_) {
        free_ = std::make_unique<Free>();
        unique_queue<DefSet> queue;

        auto enqueue = [&](const Def* def) {
            if (def->is_const()) return;

            if (auto var = def->isa<Var>())
                free_->vars.emplace(var);
            else if (auto nom = def->isa_nom())
                free_->noms.emplace(nom);
            else
                queue.push(def);
        };

        for (auto def : free_defs())
            enqueue(def);

        while (!queue.empty()) {
            for (auto op : queue.pop()->extended_ops())
                enqueue(op);
        }
    }

    return *free_;
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
    return scope.contains(def);
}

}
