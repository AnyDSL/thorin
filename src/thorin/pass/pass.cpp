#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/analyses/depends.h"

namespace thorin {

static bool needs_rewrite(Def* nom) {
    return std::any_of(nom->extended_ops().begin(), nom->extended_ops().end(), [&](const Def* op) { return op->isa<Rewrite>(); });
}

void PassMan::run() {
    world().ILOG("run");
    for (auto&& pass : passes_)
        world().ILOG(" + {}", pass->name());

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    auto externals = world().externals(); // copy
    for (const auto& [_, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");
        rewrite(old_nom)->make_external();
        old_nom->unset();
        old_nom->make_internal();
    }

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    world().ILOG("finished");
    cleanup(world());
}

Def* PassMan::inspect(Def* old_nom) {
    auto new_type  = rewrite(old_nom->type() );
    auto new_debug = rewrite(old_nom->debug() );
    auto new_nom   = old_nom->stub(world(), new_type, new_debug);

    for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i)
        new_nom->set(i, world().rewrite(old_nom->op(i), old_nom->param(), new_nom->param()));

    for (auto&& pass : passes_) {
        if (pass->enter(new_nom))
            pass->inspect(new_nom);
    }

    return new_nom;
}

const Def* PassMan::rewrite(const Def* def) {
    if (def) {
        if (auto rw = def->isa<Rewrite>())
            return rewrite(rw->def(), rw->repls());
    }
    return def;
}

Def* PassMan::rewrite(Def* nom) {
    if (!needs_rewrite(nom)) return nom;

    auto old_type  = nom->type();
    auto old_debug = nom->debug();
    auto old_ops   = nom->ops();

    Array<PassBase*> passes(passes_.size());
    size_t i = 0;
    for (auto&& pass : passes_) {
        if (pass->enter(nom)) passes[i++] = pass.get();
    }
    passes.shrink(i);

    auto new_type  = rewrite(old_type );
    auto new_debug = rewrite(old_debug);
    Array<const Def*> new_ops(old_ops, [&](const Def* def) { return rewrite(def); });

    for (auto op : nom->extended_ops()) {
        if (!analyze(op))
            return false;
    }
}

const Def* PassMan::rewrite(const Def* old_def, std::pair<const ReplArray, Def2Def>& repls) {
    if (old_def->is_const())                         return old_def;
    if (auto repl    = repls.first .find  (old_def)) return repl->replacer;
    if (auto new_def = repls.second.lookup(old_def)) return *new_def;

    if (auto rw = old_def->isa<Rewrite>()) {
        auto [it, succ] = local_.map.emplace(ReplArray(repls.first, rw->repls()), Def2Def());
        return rewrite(rw->def(), *it);
    }

    auto new_type = rewrite(old_def->type(), repls);
    auto new_dbg  = old_def->debug() ? rewrite(old_def->debug(), repls) : nullptr;

    //if (auto old_nom = old_def->isa_nominal()) {
        //if (depends(old_nom, rw->replacee()))
        //if (true)
            //return local_.map[old_def] = stub(old_nom, new_type, new_dbg);
        //return local_.map[rw] = old_nom;
    //}

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i), repls); });

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
    for (auto pass : local_.cur_passes)
        new_def = pass->rewrite(new_def);

    //if (old_def != new_def) {
        //world().DLOG("rewrite: {} -> {}", old_def, new_def);
        //local_map(old_def, new_def);
    //}

    world_.DLOG("return: {} -> {}", old_def, new_def);
    //return local_.map[old_def] = new_def;
    return nullptr;
}

bool PassMan::analyze(const Def* def) {
    if (def->is_const() || !local_.analyzed.emplace(def).second) return true;

    if (auto nom = def->isa_nominal()) {
        /*
        if (local_.old_entry == nullptr)
            global_.noms.push(nom);
        else if (outside(nom))
            local_.free.emplace(nom);
        else
        */
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
