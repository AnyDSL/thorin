#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

Def* PassMan::stub(Def* old_nom) {
    if (auto new_nom = stubs_.lookup(old_nom)) return *new_nom;
    auto new_dbg = old_nom->debug() ? lookup(old_nom->debug()) : nullptr;
    return stubs_[old_nom] = old_nom->stub(world(), lookup(old_nom->type()), new_dbg);
}

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
        auto new_nom = stub(old_nom);
        old_nom->make_internal();
        new_nom->make_external();
    }

    while (!defs.empty() || !noms.empty()) {
        while (!defs.empty()) {
            auto old_def = defs.pop();

            push(old_def->type());
            if (old_def->debug()) push(old_def->debug());

            if (auto old_nom = old_def->isa_nominal()) {
                auto new_nom = stub(old_nom);
                global_map(old_nom, new_nom);
                noms.push(old_nom);
            } else {
                for (auto op : old_def->ops()) push(op);
            }
        }

        while (!noms.empty()) {
            old_entry_ = noms.front();
            new_entry_ = lookup(old_entry_);

            if (!old_entry_->is_set()) {
                noms.pop();
                continue;
            }

            Scope s(old_entry_);
            old_scope_ = &s;

            scope_map_.clear();
            analyzed_.clear();
            passes_mask_.clear();

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
                continue;
            }

            s.visit({}, {}, {}, {}, [&](const Def* def) { push(def); });
        }
    }

    for (const auto& [name, old_nom] : externals)
        old_nom->unset();

    world().ILOG("PassMan done");
    cleanup(world_);
}

bool PassMan::scope() {
    world_.DLOG("scope: {}/{} (old_entry_/new_entry_)", old_entry_, new_entry_);
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        auto push = [&](const Def* def) {
            if (def->is_const() || def->isa_nominal() || !old_scope_->contains(def)) return false;
            return defs.push(def);
        };

        bool todo = false;
        for (auto op : def->extended_ops()) todo |= push(op);
        return todo;
    };

    noms_.push(old_entry_);

    while (!noms_.empty()) {
        old_nom_ = pop(noms_);
        new_nom_ = lookup(old_nom_);
        world_.DLOG("enter: {}/{} (old_nom_/new_nom_)", old_nom_, new_nom_);

        auto old_mask = passes_mask_;
        for (size_t i = 0, e = num_passes(); i != e; ++i) {
            if (!passes_[i]->enter(new_nom_))
                passes_mask_.clear(i);
        }

        push(old_nom_);

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

        Array<const Def*> new_ops(old_nom_->num_ops(), [&](size_t i) { return lookup(old_nom_->op(i)); });
        new_nom_->set(new_ops);

        for (auto op : new_nom_->extended_ops()) {
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

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !analyzed_.emplace(def).second) return true;

    if (auto old_nom = def->isa_nominal()) {
        auto new_nom = old_nom;

        if (old_scope_->contains(old_nom)) {
            new_nom = stub(old_nom);
            new_nom->set(old_nom->ops());
            scope_map(old_nom, new_nom);
            scope_map(old_nom->param(), new_nom->param());
        }

        foreach_pass([&](auto pass) { new_nom = pass->inspect(new_nom); });
        noms_.push(new_nom);
        return true;
    }

    bool result = true;
    for (auto op : def->extended_ops())
        result &= analyze(op);

    for (size_t i = 0, e = num_passes(); i != e; ++i) {
        if (passes_mask_[i])
            result &= passes_[i]->analyze(def);
    }

    return result;
}

}
