#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

bool Pass::enter(Scope& scope) { return scope.entry()->isa<Lam>(); }

Def* PassMan::stub(Def* old_nom) {
    if (auto new_nom = stubs_.lookup(old_nom)) return *new_nom;
    auto new_dbg = old_nom->debug() ? lookup(old_nom->debug()) : nullptr;
    return stubs_[old_nom] = old_nom->stub(world(), lookup(old_nom->type()), new_dbg);
}

void PassMan::run() {
    unique_queue<NomSet> noms;
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        if (!def->is_const()) {
            if (auto old_nom = def->isa_nominal()) {
                auto new_nom = stub(old_nom);
                global_map(old_nom, new_nom);
                noms.push(old_nom);
            } else {
                defs.push(def);
            }
        }
    };

    auto externals = world().externals(); // copy

    for (const auto& [name, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");
        push(old_nom);
        push(old_nom->type());
        push(old_nom->debug());
    }

    while (!defs.empty() || !noms.empty()) {
        while (!defs.empty()) {
            for (auto op : defs.pop()->extended_ops()) push(op);
        }

        while (!noms.empty()) {
            auto old_entry = noms.front();

            if (!old_entry->is_set()) {
                noms.pop();
                continue;
            }

            Scope scope(old_entry);
            old_scope_ = &scope;

            //scope_passes_.clear();
            //for (auto& pass : passes_) {
                //if (pass->enter(scope)) scope_passes_.push_back(pass.get());
            //}

            if (enter(old_entry)) {
                noms.pop();
                world().DLOG("done: {}", old_entry);
            } else {
                world().DLOG("retry: {}", old_entry);
                for (auto& pass : passes_) pass->retry();
                continue;
            }

            scope.visit({}, {}, {}, {}, [&](const Def* def) { push(def); });
        }
    }

    for (const auto& [name, old_nom] : externals) {
        old_nom->make_internal();
        old_nom->unset();
        lookup(old_nom)->as_nominal()->make_external();
    }

    cleanup(world_);
}

bool PassMan::enter(Def* old_entry) {
    scope_old2new_.clear();

    //Scope new_scope(new_entry);
    //new_scope_ = &new_scope;

    unique_stack<DefSet> defs;
    std::queue<Def*> noms;
    NomSet done;

    auto push = [&](const Def* def) {
        auto push = [&](const Def* def) {
            if (!def->is_const()) {
                if (old_scope().contains(def)) {
                    if (auto old_nom = def->isa_nominal()) {
                        if (done.emplace(old_nom).second) {
                            noms.push(old_nom);
                            //for (auto& pass : scope_passes_) pass->inspect(nom);
                            scope_map(old_nom, stub(old_nom));
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

    noms.push(old_entry);

    while (!noms.empty()) {
        auto old_nom = pop(noms);

        //for (auto pass : scope_passes_) pass->enter(new_nom);

        world().DLOG("{}", old_nom);

        push(old_nom);

        while (!defs.empty()) {
            auto old_def = defs.top();

            if (!push(old_def)) {
                auto new_type = lookup(old_def->type());
                auto new_dbg  = old_def->debug() ? lookup(old_def->debug()) : old_def->debug();
                Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return lookup(old_def->op(i)); });
                auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

                //for (auto pass : scope_passes_) new_def = pass->rewrite(new_def);
                scope_map(old_def, new_def);
                defs.pop();
            }
        }

        Array<const Def*> new_ops(old_nom->num_ops(), [&](size_t i) { return lookup(old_nom->op(i)); });
        lookup(old_nom)->as_nominal()->set(new_ops);
    }

    return true;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || def->isa_nominal() || !new_scope().contains(def) || analyzed_.emplace(def).second) return true;

    for (auto op : def->extended_ops()) analyze(op);
    for (auto pass : scope_passes_) {
        if (!pass->analyze(def)) return false;
    }

    return true;
}

}
