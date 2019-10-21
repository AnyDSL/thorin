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

            Scope scope(old_entry_);
            old_scope_ = &scope;

            scope_old2new_.clear();
            scope_passes_.clear();
            new_scope_ = nullptr;
            analyzed_.clear();

            for (auto& pass : passes_) {
                if (pass->enter_scope(new_entry_)) {
                    pass->clear();
                    scope_passes_.push_back(pass.get());
                }
            }

            if (enter()) {
                noms.pop();
                world().DLOG("done: {}", old_entry_);
            } else {
                world().DLOG("retry: {}", old_entry_);
                for (auto& pass : passes_) pass->retry();
                continue;
            }

            scope.visit({}, {}, {}, {}, [&](const Def* def) { push(def); });
        }
    }

    for (const auto& [name, old_nom] : externals)
        old_nom->unset();

    world().ILOG("PassMan done");
    cleanup(world_);
}

bool PassMan::enter() {
    unique_stack<DefSet> defs;
    std::queue<Def*> noms;
    NomSet done;

    auto push = [&](const Def* def) {
        auto push = [&](const Def* def) {
            if (!def->is_const()) {
                if (old_scope_->contains(def)) {
                    if (auto old_nom = def->isa_nominal()) {
                        if (done.emplace(old_nom).second) {
                            noms.push(old_nom);
                            auto new_nom = stub(old_nom);
                            scope_map(old_nom, new_nom);
                            for (auto& pass : scope_passes_) pass->inspect(new_nom);
                            return true;
                        }
                    } else {
                        return defs.push(def);
                    }
                }
            }
            return false;
        };

        bool todo = false;
        for (auto op : def->extended_ops()) todo |= push(op);
        return todo;
    };

    noms.push(old_entry_);

    while (!noms.empty()) {
        old_nom_ = pop(noms);
        new_nom_ = lookup(old_nom_);

        nom_passes_.clear();
        for (auto pass : scope_passes_) {
            if (pass->enter_nominal(new_nom_)) nom_passes_.push_back(pass);
        }

        push(old_nom_);

        while (!defs.empty()) {
            auto old_def = defs.top();

            if (!push(old_def)) {
                auto new_type = lookup(old_def->type());
                auto new_dbg  = old_def->debug() ? lookup(old_def->debug()) : nullptr;
                Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return lookup(old_def->op(i)); });
                auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

                for (auto pass : nom_passes_) new_def = pass->rewrite(new_def);
                scope_map(old_def, new_def);
                defs.pop();
            }
        }

        Array<const Def*> new_ops(old_nom_->num_ops(), [&](size_t i) { return lookup(old_nom_->op(i)); });
        new_nom_->set(new_ops);

        if (new_scope_)
            new_scope_->update();
        else
            new_scope_ = std::make_unique<Scope>(new_entry_);

        for (auto op : new_nom_->extended_ops()) {
            if (!analyze(op)) {
                for (auto& pass : nom_passes_) pass->retry();
                return false;
            }
        }
    }

    return true;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || def->isa_nominal() || !new_scope().contains(def) || !analyzed_.emplace(def).second) return true;
    world().DLOG("analyze: {}", def);

    for (auto op : def->extended_ops()) analyze(op);
    for (auto pass : nom_passes_) {
        if (!pass->analyze(def)) return false;
    }

    return true;
}

}
