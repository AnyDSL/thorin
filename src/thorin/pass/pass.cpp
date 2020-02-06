#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/analyses/depends.h"

namespace thorin {

Def* PassMan::stub(Def* old_nom, const Def* type, const Def* dbg) {
    Def* new_nom;

    auto [i, success] = stubs_.emplace(old_nom, nullptr);
    if (success) {
        new_nom = i->second = old_nom->stub(world(), type, dbg);
    } else {
        new_nom = i->second;
    }

    if (old_nom->is_set()) {
        for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i)
            new_nom->set(i, world().rewrite(old_nom->op(i), old_nom->param(), new_nom->param()));
    }

    return new_nom;
}

bool PassMan::outside(Def*) {
    // TODO
    return true;
}

void PassMan::run() {
    world().ILOG("run");
    for (auto&& pass : passes_)
        world().ILOG(" + {}", pass->name());

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    Def* hack_old;
    Def* hack_new;
    auto externals = world().externals(); // copy
    for (const auto& [_, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");
        hack_old = old_nom;
        hack_new = stub(old_nom, old_nom->type(), old_nom->debug()); // TODO type, debug
        global_.noms.push(hack_new);
    }

    while (!global_.noms.empty()) {
        local_.new_entry = global_.noms.front();
        local_.old_entry = local_.new_entry->ops().back()->as<Rewrite>()->old_def()->as<Param>()->nominal(); // TODO this is a bit hacky

        if (!local_.new_entry->is_set()) {
            global_.noms.pop();
            continue;
        }

        Array<const Def*> prev_ops = local_.new_entry->ops();
        local_.clear();

        for (auto&& pass : passes_) {
            if (pass->scope(local_.new_entry))
                local_.passes.emplace_back(pass.get());
        }

        if (scope()) {
            global_.noms.pop();
            world().DLOG("done: {}", local_.old_entry);
            for (auto pass : local_.passes) pass->clear();
        } else {
            world().DLOG("retry: {}", local_.old_entry);
            for (auto& pass : local_.passes) pass->retry();
            local_.new_entry->set(prev_ops);
            continue;
        }

        for (auto nom : local_.free) global_.noms.push(nom);
    }

#if 0
    for (const auto& [name, old_nom] : externals) {
        old_nom->unset();
        old_nom->make_internal();
        lookup(old_nom)->make_external();
    }
#endif
    hack_old->unset();
    hack_old->make_internal();
    hack_new->make_external();

    world().ILOG("finished");
    cleanup(world());
}

bool PassMan::scope() {
    world_.DLOG("scope: {}/{} (old_entry/new_entry)", local_.old_entry, local_.new_entry);
    local_.noms.push(local_.new_entry);

    while (!local_.noms.empty()) {
        for (size_t i = 0, e = local_.old_entry->num_ops(); i != e; ++i)
            assert(local_.old_entry->op(i) == local_.new_entry->op(i)->as<Rewrite>()->def());

        cur_nom_ = local_.noms.pop();
        world_.DLOG("enter: {} (cur_nom)", cur_nom_);

        local_.cur_passes.clear();
        for (auto pass : local_.passes) {
            if (pass->enter(cur_nom_))
                local_.cur_passes.emplace_back(pass);
        }

        Array<const Def*> new_ops(cur_nom_->num_ops(), [&](size_t i) { return rewrite(cur_nom_->op(i)->as<Rewrite>()); });
        cur_nom_->set(new_ops);

        for (auto op : cur_nom_->extended_ops()) {
            if (!analyze(op))
                return false;
        }
    }

    return true;
}

const Def* PassMan::wrap_rewrite(const Def* def, const Def* old_def, const Def* new_def) {
    if (def->is_const()) return def;
    return rewrite(world().rewrite(def, old_def, new_def));
}

const Def* PassMan::rewrite(const Rewrite* rw) {
    if (auto new_def = local_.map.lookup(rw)) return *new_def;

    auto new_type = wrap_rewrite(rw->def()->type(), rw->old_def(), rw->new_def());
    auto new_dbg  = rw->def()->debug() ? wrap_rewrite(rw->def()->debug(), rw->old_def(), rw->new_def()) : nullptr;

    if (auto old_nom = rw->def()->isa_nominal()) {
        if (depends(old_nom, rw->old_def()))
            return local_.map[rw] = stub(old_nom, new_type, new_dbg);
        return local_.map[rw] = old_nom;
    }

#if 0
    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(rw->op(i)); });

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
    for (auto pass : local_.cur_passes)
        new_def = pass->rewrite(new_def);

    if (old_def != new_def) {
        world().DLOG("rewrite: {} -> {}", old_def, new_def);
        local_map(old_def, new_def);
    }

    return old_def;

    if (old_def->is_const()) return old_def;
    if (auto new_def = lookup(old_def); new_def != old_def) return new_def;

    if (auto old_nom = old_def->isa_nominal()) {
        if (new2old_.contains(old_nom)) return old_nom;

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
            world().DLOG("global inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
            //for (auto&& pass : passes_) new_nom = pass->global_inspect(new_nom);
        } else {
            local_map(old_nom, new_nom);
            local_map(old_nom->param(), new_nom->param());

            world().DLOG("local inspect: {}/{} (old_nom/new_nom)", old_nom, new_nom);
            // Pass through the inspected nominal but return the original new_nom.
            // The passes must take care of the inspected nominal by themselves.
            auto new_inspected = new_nom;
            for (auto pass : local_.passes)
                new_inspected = pass->inspect(new_inspected);
        }

        world().DLOG("new2old_: {}/{} (old_nom/new_nom)", old_nom, new_nom);
        new2old_[new_nom] = old_nom;
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
#endif
    return rw;
}

bool PassMan::analyze(const Def* def) {
    return true;
    if (def->is_const() || !local_.analyzed.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        if (local_.old_entry == nullptr)
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
