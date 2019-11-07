#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

/*
 * helpers
 */

bool PassMan::within(Def* nom) {
    return !nom->is_const() && !old_scope_free_->contains(nom) && !ops2old_entry_.contains(nom->ops());
}

Def* PassMan::stub(Def* old_nom) {
    Def* new_nom;

    if (auto cached = stubs_.lookup(old_nom)) {
        new_nom = *cached;
    } else {
        auto new_dbg = old_nom->debug() ? lookup(old_nom->debug()) : nullptr;
        new_nom = old_nom->stub(world(), lookup(old_nom->type()), new_dbg);
        stubs_[old_nom] = new_nom;
    }

    new_nom->set(old_nom->ops());
    return new_nom;
}

Def* PassMan::global_stub(Def* old_nom) {
    auto new_nom = stub(old_nom);
    ops2old_entry_[old_nom->ops()] = old_nom;
    global_map(old_nom, new_nom);
    return new_nom;
}

Def* PassMan::scope_stub(Def* old_nom) {
    auto new_nom = stub(old_nom);
    scope_map(old_nom, new_nom);
    scope_map(old_nom->param(), new_nom->param());
    return new_nom;
}

/*
 * main driver that enters all found top-level scopes
 */

void PassMan::run() {
    world().ILOG("PassMan start");

    unique_queue<NomSet> noms;
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        if (!def->is_const()) defs.push(def);
    };

    auto externals = world().externals(); // copy
    for (const auto& [name, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");
        defs.push(old_nom);
    }

    while (!defs.empty() || !noms.empty()) {
        while (!defs.empty()) {
            auto old_def = defs.pop();

            push(old_def->type());
            if (old_def->debug()) push(old_def->debug());

            if (auto old_nom = old_def->isa_nominal()) {
                noms.push(global_stub(old_nom));
            } else {
                for (auto op : old_def->ops()) push(op);
            }
        }

        while (!noms.empty()) {
            new_entry_ = noms.front();

            if (!new_entry_->is_set()) {
                noms.pop();
                continue;
            }

            old_entry_ = ops2old_entry_[new_entry_->ops()];
            Scope s(old_entry_);
            old_scope_ = &s;
            old_scope_free_ = &old_scope_->free();

            passes_mask_.clear();
            scope_map_  .clear();
            analyzed_   .clear();
            scope_noms_ .clear();
            inspected_  .clear();
            free_noms_  .clear();

            for (size_t i = 0, e = num_passes(); i != e; ++i) {
                if (passes_[i]->scope(new_entry_))
                    passes_mask_.set(i);
            }

            if (scope()) {
                noms.pop();
                world().DLOG("done: {}", old_entry_);
                foreach_pass([&](auto pass) { pass->clear(); });
            } else {
                world().DLOG("retry: {}", old_entry_);
                for (auto& pass : passes_) pass->retry();
                new_entry_->set(old_entry_->ops());
                continue;
            }

            for (auto nom : free_noms_) noms.push(nom);
        }
    }

    for (const auto& [name, old_nom] : externals) {
        old_nom->unset();
        old_nom->make_internal();
        lookup(old_nom)->make_external();
    }

    world().ILOG("PassMan done");
    cleanup(world_);
}

/*
 * processes one top-level scope
 */

bool PassMan::scope() {
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        auto push = [&](const Def* def) {
            if (def->is_const() || old_scope_free_->contains(def)) return false;

            if (auto old_nom = def->isa_nominal()) {
                if (!ops2old_entry_.contains(old_nom->ops()) && inspected_.emplace(old_nom).second) {
                    auto new_nom = scope_stub(old_nom);
                    world().DLOG("inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                    foreach_pass([&](auto pass) { new_nom = pass->inspect(new_nom); });
                }
                return false;
            }

            return defs.push(def);
        };

        bool todo = false;
        for (auto op : def->extended_ops()) todo |= push(op);
        return todo;
    };

    world_.DLOG("scope: {}/{} (old_entry_/new_entry_)", old_entry_, new_entry_);
    scope_map(old_entry_->param(), new_entry_->param());
    scope_noms_.push(new_entry_);

    while (!scope_noms_.empty()) {
        cur_nom_ = scope_noms_.pop();
        world_.DLOG("enter: {} (cur_nom)", cur_nom_);

        auto old_mask = passes_mask_;
        for (size_t i = 0, e = num_passes(); i != e; ++i) {
            if (!passes_[i]->enter(cur_nom_))
                passes_mask_.clear(i);
        }

        push(cur_nom_);

        while (!defs.empty()) {
            auto old_def = defs.top();

            if (!push(old_def)) {
                if (old_def == lookup(old_def)) {
                    auto new_type = lookup(old_def->type());
                    auto new_dbg  = old_def->debug() ? lookup(old_def->debug()) : nullptr;
                    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return lookup(old_def->op(i)); });

                    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
                    foreach_pass([&](auto pass) { new_def = pass->rewrite(new_def); });
                    if (old_def != new_def) {
                        world().DLOG("rewrite: {} -> {}", old_def, new_def);
                        scope_map(old_def, new_def);
                    }
                }
                defs.pop();
            }
        }

        Array<const Def*> new_ops(cur_nom_->num_ops(), [&](size_t i) { return lookup(cur_nom_->op(i)); });
        cur_nom_->set(new_ops);

        for (auto op : cur_nom_->extended_ops()) {
            if (!analyze(op)) {
                passes_mask_ = old_mask;
                foreach_pass([&](auto pass) { pass->retry(); });
                return false;
            }
        }

        passes_mask_ = old_mask;
    }

    return true;
}

/*
 * analyze cur_nom_ in current scope
 */

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || old_scope_free_->contains(def) || !analyzed_.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        if (ops2old_entry_.contains(nom->ops()))
            free_noms_.emplace(nom);
        else
            scope_noms_.push(nom);
        return true;
    }

    bool result = true;
    for (auto op : def->extended_ops())
        result &= analyze(op);

    for (size_t i = 0, e = num_passes(); i != e; ++i) {
        world().DLOG("analyze: {}", def);
        if (passes_mask_[i])
            result &= passes_[i]->analyze(def);
    }

    return result;
}

}
