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
    free_      = nullptr;
    free_vars_ = nullptr;
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

const DefSet& Scope::free() const {
    if (!free_) {
        free_ = std::make_unique<DefSet>();

        for (auto def : defs_) {
            for (auto op : def->extended_ops()) {
                if (!op->is_const() && !contains(op))
                    free_->emplace(op);
            }
        }
    }

    return *free_;
}

const VarSet& Scope::free_vars() const {
    if (!free_vars_) {
        free_vars_ = std::make_unique<VarSet>();
        unique_queue<DefSet> queue;

        auto enqueue = [&](const Def* def) {
            if (auto var = def->isa<Var>())
                free_vars_->emplace(var);
            else if (def->isa_nominal())
                return;
            else
                queue.push(def);
        };

        for (auto def : free())
            enqueue(def);

        while (!queue.empty()) {
            for (auto op : queue.pop()->extended_ops())
                enqueue(op);
        }
    }

    return *free_vars_;
}

const NomSet& Scope::free_noms() const {
    if (!free_noms_) {
        free_noms_ = std::make_unique<NomSet>();
        unique_queue<DefSet> queue;

        auto enqueue = [&](const Def* def) {
            if (auto nom = def->isa_nominal())
                free_noms_->emplace(nom);
            else
                queue.push(def);
        };

        for (auto def : free())
            enqueue(def);

        while (!queue.empty()) {
            for (auto op : queue.pop()->extended_ops())
                enqueue(op);
        }
    }

    return *free_noms_;
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
    for (auto p : var->nominal()->vars())
        if (p == var) return true;

    Scope scope(var->nominal());
    return scope.contains(def);
}

}
