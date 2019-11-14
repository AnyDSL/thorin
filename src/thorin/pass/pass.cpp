#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

bool PassMan::outside(Def* nom) {
    if (!nom->is_set()) return true;
    if (auto old_nom = ops2old_entry_.lookup(nom->ops())) nom = *old_nom;
    return old_scope_->free_noms().contains(nom);
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

Def* PassMan::local_stub(Def* old_nom) {
    auto new_nom = stub(old_nom);
    local_map(old_nom, new_nom);
    local_map(old_nom->param(), new_nom->param());
    return new_nom;
}

void PassMan::run() {
    world().ILOG("run");

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
            local_.clear();

            for (auto&& pass : passes_) {
                if (pass->scope(new_entry_))
                    local_.passes_.emplace_back(pass.get());
            }

            if (scope()) {
                noms.pop();
                world().DLOG("done: {}", old_entry_);
                for (auto pass : local_.passes_) pass->clear();
            } else {
                world().DLOG("retry: {}", old_entry_);
                for (auto& pass : local_.passes_) pass->retry();
                new_entry_->set(old_entry_->ops());
                continue;
            }

            for (auto nom : local_.free_) noms.push(nom);
        }
    }

    for (const auto& [name, old_nom] : externals) {
        old_nom->unset();
        old_nom->make_internal();
        lookup(old_nom)->make_external();
    }

    world().ILOG("finished");
    cleanup(world_);
}

bool PassMan::scope() {
    world_.DLOG("scope: {}/{} (old_entry_/new_entry_)", old_entry_, new_entry_);
    local_map(old_entry_->param(), new_entry_->param());
    local_.noms_.push(new_entry_);

    while (!local_.noms_.empty()) {
        cur_nom_ = local_.noms_.pop();
        world_.DLOG("enter: {} (cur_nom)", cur_nom_);

        local_.cur_passes_.clear();
        for (auto pass : local_.passes_) {
            if (pass->enter(cur_nom_)) local_.cur_passes_.emplace_back(pass);
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
    if (!local_.rewritten_.emplace(old_def).second) return lookup(old_def);
    if (auto cached = lookup(old_def); cached != old_def) return cached;

    if (auto old_nom = old_def->isa_nominal()) {
        if (outside(old_nom)) {
            if (global_.inspected_.emplace(old_nom).second) {
                auto new_nom = global_stub(old_nom);
                auto success = global_.inspected_.emplace(new_nom).second;
                assert(success);
                world().DLOG("global inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                //for (auto&& pass : passes_) new_nom = pass->global_inspect(new_nom);
                return new_nom;
            }
        } else {
            if (local_.inspected_.emplace(old_nom).second) {
                auto new_nom = local_stub(old_nom);
                auto success = local_.inspected_.emplace(new_nom).second;
                assert(success);
                world().DLOG("local inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
                for (auto pass : local_.passes_)
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
    for (auto pass : local_.cur_passes_)
        new_def = pass->rewrite(new_def);

    if (old_def != new_def) {
        world().DLOG("rewrite: {} -> {}", old_def, new_def);
        local_map(old_def, new_def);
    }

    return new_def;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !local_.analyzed_.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        if (outside(nom))
            local_.free_.emplace(nom);
        else
            local_.noms_.push(nom);
        return true;
    }

    bool result = true;
    for (auto op : def->extended_ops())
        result &= analyze(op);

    for (auto pass : local_.passes_) {
        world().DLOG("analyze: {}", def);
        result &= pass->analyze(def);
    }

    return result;
}

}
