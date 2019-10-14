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

void Scope::visit(VisitNomFn pre_nom, VisitDefFn pre_def, VisitDefFn post_def, VisitNomFn post_nom, VisitDefFn free) {
    unique_queue<NomSet> noms;
    unique_stack<DefSet> defs;

    noms.push(entry());

    auto push = [&](const Def* def) {
        if (contains(def)) {
            if (auto nom = def->isa_nominal())
                noms.push(nom);
            else
                return defs.push(def);
        }
        if (free) free(def);
        return false;
    };

    while (!noms.empty()) {
        auto nom = noms.pop();
        if (pre_nom) pre_nom(nom);
        for (auto op : nom->ops()) push(op);

        while (!defs.empty()) {
            auto def = defs.top();

            if (pre_def) pre_def(def);

            bool todo = false;
            for (auto op : def->ops())
                todo |= push(op);

            if (!todo) {
                if (post_def) post_def(def);
                defs.pop();
            }
        }

        if (post_nom) post_nom(nom);
    }
}

bool Scope::rewrite(const std::string& info, RewriteFn fn) {
    world().VLOG("{}: rewriting scope {} - start", info, entry());

    Def2Def old2new;
    bool dirty = false;

    auto make_new_ops = [&](const Def* old_def) {
        return Array<const Def*>(old_def->num_ops(), [&](size_t i) {
            if (auto new_def = old2new.lookup(old_def->op(i))) return *new_def;
            return old_def->op(i);
        });
    };

    visit({}, {},
        [&](const Def* old_def) {
            if (auto new_def = fn(old_def)) {
                old2new[old_def] = new_def;
            } else {
                old2new[old_def] = old_def->rebuild(world(), old_def->type(), make_new_ops(old_def), old_def->debug());
            }
        },
        [&](Def* old_nom) {         // post-order nominmals
            auto new_ops = make_new_ops(old_nom);

            if (!std::equal(new_ops.begin(), new_ops.end(), old_nom->ops().begin())) {
                old_nom->set(new_ops);
                dirty = true;
            }
        });

    world().VLOG("{}: rewriting scope {} - done,", info, entry());
    return dirty;
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

Stream& Scope::stream(Stream& s) const { return schedule(*this).stream(s); }

template void Streamable<Scope>::dump() const;
template void Streamable<Scope>::write() const;

}
