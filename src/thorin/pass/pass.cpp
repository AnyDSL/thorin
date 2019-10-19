#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

bool Pass::enter(Scope& scope) { return scope.entry()->isa<Lam>(); }

Def* PassMan::stub(Def* old_nom) {
    if (auto new_nom = stubs_.lookup(old_nom)) return *new_nom;
    return stubs_[old_nom] = old_nom->stub(world(), old_nom->type(), old_nom->debug());
}

void PassMan::run() {
    unique_queue<NomSet> noms;
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        if (!def->is_const()) {
            if (auto nom = def->isa_nominal())
                noms.push(nom);
            else
                defs.push(def);
        }
    };

    for (const auto& [name, nom] : world().externals()) {
        assert(nom->is_set() && "external must not be empty");
        noms.push(nom);
    }

    while (!noms.empty()) {
        auto nom = noms.front();
        if (!nom->is_set()) continue;
        Scope scope(nom);

        scope_passes_.clear();
        for (auto& pass : passes_) {
            if (pass->enter(scope)) scope_passes_.push_back(pass.get());
        }

        if (auto new_nom = run(scope)) {
            nom->set(new_nom->ops());
            new_nom->unset();
            noms.pop();
        } else {
            for (auto& pass : passes_) pass->retry();
            continue;
        }

        scope.visit({}, {}, {}, {}, [&](const Def* def) { push(def); });

        while (!defs.empty()) {
            for (auto op : defs.pop()->extended_ops()) push(op);
        }
    }

    cleanup(world_);
}

Def* PassMan::run(Scope& scope) {
    scope_ = &scope;

    Def2Def old2new;

    auto lookup = [&](const Def* old_def) {
        if (auto new_def = old2new.lookup(old_def)) return *new_def;
        return old_def;
    };

    auto make_new_ops = [&](const Def* def, bool& changed) {
        return Array<const Def*>(def->num_ops(), [&](size_t i) {
            auto new_op = lookup(def->op(i));
            changed |= def->op(i) != new_op;
            return new_op;
        });
    };

    std::queue<Def*> noms;
    NomSet done;
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        auto push = [&](const Def* def) {
            if (!def->is_const()) {
                if (scope.contains(def)) {
                    if (auto nom = def->isa_nominal()) {
                        if (done.emplace(nom).second) {
                            noms.push(nom);
                            for (auto& pass : scope_passes_) pass->inspect(nom);
                        }
                        return false;
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

    noms.push(scope.entry());

    while (!noms.empty()) {
        auto old_nom = pop(noms);
        auto new_nom = stub(old_nom);
        old2new[old_nom] = new_nom;

        for (auto pass : scope_passes_) pass->enter(new_nom);

        push(old_nom);

        while (!defs.empty()) {
            auto def = defs.top();

            if (!push(def)) {
                auto new_type = lookup(def->type());
                auto new_dbg  = lookup(def->debug());

                bool changed = false;
                changed |= new_type != def->type();
                changed |= new_dbg  != def->debug();
                auto new_ops = make_new_ops(def, changed);
                auto new_def = changed ? def->rebuild(world(), new_type, new_ops, def->debug()) : def;

                for (auto pass : scope_passes_) new_def = pass->rewrite(new_def);
                old2new[def] = new_def;
                defs.pop();
            }
        }

        bool changed = false;
        auto new_ops = make_new_ops(old_nom, changed);
        new_nom->set(new_ops);
        for (auto op : new_nom->extended_ops()) {
            if (!analyze(op)) return nullptr;
        }
    }

    return stub(scope.entry());
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !scope().contains(def) || analyzed_.emplace(def).second) return true;

    for (auto op : def->ops()) analyze(op);
    for (auto pass : scope_passes_) {
        if (!pass->analyze(def)) return false;
    }

    return true;
}

}
