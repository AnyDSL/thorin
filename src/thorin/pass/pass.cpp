#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

bool PassMan::outside(Def* nom) {
    if (local_.old_scope == nullptr || !nom->is_set()) return true;
    if (auto old_nom = new2old_.lookup(nom)) nom = *old_nom;
    return local_.old_scope->free_noms().contains(nom);
}

void PassMan::run() {
    world().ILOG("run");

    auto externals = world().externals(); // copy
    for (const auto& [name, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");
        analyze(rewrite(old_nom));
    }

    while (!global_.noms.empty()) {
        new_entry_ = global_.noms.front();

        if (!new_entry_->is_set()) {
            global_.noms.pop();
            continue;
        }

        old_entry_ = new2old_[new_entry_];
        assert(old_entry_ && old_entry_ != new_entry_);
        Scope s(old_entry_);
        local_.clear(s);

        for (auto&& pass : passes_) {
            if (pass->scope(new_entry_))
                local_.passes.emplace_back(pass.get());
        }

        if (scope()) {
            global_.noms.pop();
            world().DLOG("done: {}", old_entry_);
            for (auto pass : local_.passes) pass->clear();
        } else {
            world().DLOG("retry: {}", old_entry_);
            for (auto& pass : local_.passes) pass->retry();
            new_entry_->set(old_entry_->ops());
            continue;
        }

        local_.old_scope = nullptr;
        for (auto nom : local_.free) global_.noms.push(nom);
    }

    for (const auto& [name, old_nom] : externals) {
        old_nom->unset();
        old_nom->make_internal();
        lookup(old_nom)->make_external();
    }

    world().ILOG("finished");
    cleanup(world());
}

bool PassMan::scope() {
    world_.DLOG("scope: {}/{} (old_entry_/new_entry_)", old_entry_, new_entry_);
    local_map(old_entry_->param(), new_entry_->param());
    local_.noms.push(new_entry_);

    while (!local_.noms.empty()) {
        cur_nom_ = local_.noms.pop();
        world_.DLOG("enter: {} (cur_nom)", cur_nom_);

        local_.cur_passes.clear();
        for (auto pass : local_.passes) {
            if (pass->enter(cur_nom_)) local_.cur_passes.emplace_back(pass);
        }

        Array<const Def*> new_ops(cur_nom_->num_ops(), [&](size_t i) { return rewrite(cur_nom_->op(i)); });
        cur_nom_->set(new_ops);

        for (auto op : cur_nom_->extended_ops()) {
            if (!analyze(op)) return false;
        }
    }

    return true;
}

const Def* PassMan::rewrite(const Def* old_def) {
    if (old_def->is_const()) return old_def;
    if (!local_.rewritten.emplace(old_def).second) return lookup(old_def);
    if (auto new_def = lookup(old_def); new_def != old_def) return new_def;

    auto new_type = rewrite(old_def->type());
    auto new_dbg  = old_def->debug() ? rewrite(old_def->debug()) : nullptr;

    if (auto old_nom = old_def->isa_nominal()) {
        Def* new_nom;
        if (auto cached = stubs_.lookup(old_nom)) {
            new_nom = *cached;
        } else {
            new_nom = old_nom->stub(world(), new_type, new_dbg);
            stubs_[old_nom] = new_nom;
        }
        new_nom->set(old_nom->ops());

        if (outside(old_nom)) {
            global_map(old_nom, new_nom);
            if (old_nom->is_set()) {
                world().DLOG("new2old_: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                new2old_[new_nom] = old_nom;
            }

            world().DLOG("global inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
            //for (auto&& pass : passes_) new_nom = pass->global_inspect(new_nom);
        } else {
            local_map(old_nom, new_nom);
            local_map(old_nom->param(), new_nom->param());

            world().DLOG("local inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
            for (auto pass : local_.passes)
                new_nom = pass->inspect(new_nom);
        }

        return new_nom;
    }

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i)); });

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
    for (auto pass : local_.cur_passes)
        new_def = pass->rewrite(new_def);

    if (old_def != new_def) {
        world().DLOG("rewrite: {} -> {}", old_def, new_def);
        local_map(old_def, new_def);
    }

    return new_def;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !local_.analyzed.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        if (local_.old_scope == nullptr)
            global_.noms.push(nom);
        else if (outside(nom))
            local_.free.emplace(nom);
        else
            local_.noms.push(nom);
        return true;
    }

    bool result = true;
    for (auto op : def->extended_ops())
        result &= analyze(op);

    for (auto pass : local_.passes) {
        world().DLOG("analyze: {}", def);
        result &= pass->analyze(def);
    }

    return result;
}

}
