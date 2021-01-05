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
    unique_queue<DefSet&> queue(defs_);
    queue.push(entry_->var());

    while (!queue.empty()) {
        for (auto use : queue.pop()->uses()) {
            if (use == entry_ || use == exit_) continue;
            queue.push(use);
        }
    }
}

const Scope::Free& Scope::free() const {
    if (!free_) {
        free_ = std::make_unique<Free>();
        unique_queue<DefSet> q_in;
        unique_queue<DefSet> q_out;

        auto enq_out = [&](const Def* def) {
            if (def->is_const()) return;

            if (auto var = def->isa<Var>())
                free_->vars.emplace(var);
            else if (auto nom = def->isa_nom())
                free_->noms.emplace(nom);
            else
                q_out.push(def);
        };

        auto enq_in = [&](const Def* def) {
            if (def->is_const()) return;

            if (contains(def))
                q_in.push(def);
            else {
                free_->defs.emplace(def);
                enq_out(def);
            }
        };

        for (auto op : entry()->extended_ops())
            enq_in(op);

        while (!q_in.empty()) {
            for (auto op : q_in.pop()->extended_ops())
                enq_in(op);
        }

        while (!q_out.empty()) {
            for (auto op : q_out.pop()->extended_ops())
                enq_out(op);
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
