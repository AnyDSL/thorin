#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

bool PassMan::outside(Def* nom) {
    if (!nom->is_set()) return true;
    if (auto old_nom = ops2old_entry_.lookup(nom->ops())) nom = *old_nom;
    return old_scope_free_->contains(nom);
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
    global_map(old_nom, new_nom);

    if (old_nom->is_set())
        ops2old_entry_[old_nom->ops()] = old_nom;

    return new_nom;
}

Def* PassMan::scope_stub(Def* old_nom) {
    auto new_nom = stub(old_nom);
    scope_map(old_nom, new_nom);
    scope_map(old_nom->param(), new_nom->param());
    return new_nom;
}

void PassMan::run() {
    world().ILOG("start");

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
            old_scope_free_ = &old_scope_->free_noms();

            for (auto def : *old_scope_free_)
                outf("{}: {}", def->gid(), def);

            scope_map_  .clear();
            analyzed_   .clear();
            rewritten_  .clear();
            scope_noms_ .clear();
            inspected_  .clear();
            free_noms_  .clear();

            scope_passes_.clear();
            for (auto&& pass : passes_) {
                if (pass->scope(new_entry_))
                    scope_passes_.emplace_back(pass.get());
            }

            if (scope()) {
                noms.pop();
                world().DLOG("done: {}", old_entry_);
                for (auto pass : scope_passes_) pass->clear();
            } else {
                world().DLOG("retry: {}", old_entry_);
                for (auto& pass : passes_) pass->retry();
                new_entry_->set(old_entry_->ops());
                continue;
            }

            for (auto nom : free_noms_)
                noms.push(nom);
        }
    }

    for (const auto& [name, old_nom] : externals) {
        old_nom->unset();
        old_nom->make_internal();
        lookup(old_nom)->make_external();
    }

    world().ILOG("end");
    cleanup(world_);
}

/*
 * processes one top-level scope
 */

bool PassMan::scope() {
    world_.DLOG("scope: {}/{} (old_entry_/new_entry_)", old_entry_, new_entry_);
    scope_map(old_entry_->param(), new_entry_->param());
    scope_noms_.push(new_entry_);

    while (!scope_noms_.empty()) {
        cur_nom_ = scope_noms_.pop();
        world_.DLOG("enter: {} (cur_nom)", cur_nom_);

        cur_passes_.clear();
        for (auto pass : scope_passes_) {
            if (pass->enter(cur_nom_))
                cur_passes_.emplace_back(pass);
        }

        Array<const Def*> new_ops(cur_nom_->num_ops(), [&](size_t i) { return rewrite(cur_nom_->op(i)); });
        cur_nom_->set(new_ops);

        for (auto op : cur_nom_->extended_ops()) {
            if (!analyze(op)) {
                for (auto pass : scope_passes_) pass->retry();
                return false;
            }
        }
    }

    return true;
}

const Def* PassMan::rewrite(const Def* old_def) {
    if (old_def->is_const()) return old_def;
    auto cached = lookup(old_def);
    if (cached != old_def) return cached;
    if (!rewritten_.emplace(old_def).second) return lookup(old_def);

    if (auto old_nom = old_def->isa_nominal()) {
        if (outside(old_nom)) {
            if (global_inspected_.emplace(old_nom).second) {
                auto new_nom = global_stub(old_nom);
                auto success = global_inspected_.emplace(new_nom).second;
                assert(success);
                world().DLOG("global inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                //foreach_pass([&](auto pass) { new_nom = pass->inspect(new_nom); });
                return new_nom;
            }
        } else {
            if (inspected_.emplace(old_nom).second) {
                auto new_nom = scope_stub(old_nom);
                auto success = inspected_.emplace(new_nom).second;
                assert(success);
                world().DLOG("inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                for (auto pass : scope_passes_)
                    new_nom = pass->inspect(new_nom);
                return new_nom;
            }
        }

        return lookup(old_nom);
    }

    auto new_type = rewrite(old_def->type());
    auto new_dbg  = old_def->debug() ? rewrite(old_def->debug()) : nullptr;
    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i)); });

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
    for (auto pass : cur_passes_)
        new_def = pass->rewrite(new_def);

    if (old_def != new_def) {
        world().DLOG("rewrite: {} -> {}", old_def, new_def);
        scope_map(old_def, new_def);
    }

    return new_def;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !analyzed_.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        if (outside(nom))
            free_noms_.emplace(nom);
        else
            scope_noms_.push(nom);
        return true;
    }

    bool result = true;
    for (auto op : def->extended_ops())
        result &= analyze(op);

    for (auto pass : scope_passes_) {
        world().DLOG("analyze: {}", def);
        result &= pass->analyze(def);
    }

    return result;
}

}
