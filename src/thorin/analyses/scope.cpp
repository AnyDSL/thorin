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
    , exit_(world().lam(world().cn(world().bot_star()), {"exit"}))
{
    run();
}

Scope::~Scope() {}

Scope& Scope::update() {
    defs_.clear();
    free_        = nullptr;
    free_params_ = nullptr;
    cfa_         = nullptr;
    run();
    return *this;
}

void Scope::run() {
    unique_queue<DefSet&> queue(defs_);
    queue.push(entry_->param());

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
            if (!def->is_set()) continue;

            for (auto op : def->ops()) {
                if (!contains(op))
                    free_->emplace(op);
            }
        }
    }

    return *free_;
}

const ParamSet& Scope::free_params() const {
    if (!free_) {
        free_params_ = std::make_unique<ParamSet>();
        unique_queue<DefSet> queue;

        auto enqueue = [&](const Def* def) {
            if (auto param = def->isa<Param>())
                free_params_->emplace(param);
            else if (def->isa_nominal())
                return;
            else
                queue.push(def);
        };

        for (auto def : free())
            enqueue(def);

        while (!queue.empty()) {
            for (auto op : queue.pop()->ops())
                enqueue(op);
        }
    }

    return *free_params_;
}

bool Scope::rewrite(RewriteFn pre_order_fn, RewriteFn post_order_fn) {
    unique_queue<NomSet> noms;
    unique_stack<DefSet> defs;
    Def2Def old2new;

    noms.push(entry());

    auto push = [&](const Def* def) {
        if (contains(def)) {
            if (auto nom = def->isa_nominal())
                noms.push(nom);
            else
                return defs.push(def);
        }
        return false;
    };

    bool dirty = false;
    while (!noms.empty()) {
        auto nom = noms.pop();
        for (auto op : nom->ops()) push(op);

        while (!defs.empty()) {
            auto def = defs.top();

            if (auto new_def = old2new.lookup(def)) {
                def = *new_def;
            } else {
                if (auto new_def = pre_order_fn(def))
                    def = old2new[def] = new_def;
            }

            bool todo = false;
            for (auto op : def->ops())
                todo |= push(op);

            if (!todo) {
                if (auto new_def = post_order_fn(def))
                    def = old2new[def] = new_def;

                Array<const Def*> new_ops(def->num_ops(), [&](size_t i) {
                    if (auto new_def = old2new.lookup(def->op(i))) return *new_def;
                    return def->op(i);
                });

                auto new_def = def->rebuild(world(), def->type(), new_ops, def->debug());
                old2new[def] = new_def;

                defs.pop();
            }
        }

        Array<const Def*> new_ops(nom->num_ops(), [&](size_t i) {
            if (auto new_def = old2new.lookup(nom->op(i))) return *new_def;
            return nom->op(i);
        });

        if (!std::equal(new_ops.begin(), new_ops.end(), nom->ops().begin())) {
            nom->set(new_ops);
            dirty = true;
        }
    }

    return dirty;
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

Stream& Scope::stream(Stream& s) const { return schedule(*this).stream(s); }

template void Streamable<Scope>::dump() const;
template void Streamable<Scope>::write() const;

}
