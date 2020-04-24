#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/analyses/depends.h"

namespace thorin {

static bool has_subst(Def* nom) {
    return std::any_of(nom->extended_ops().begin(), nom->extended_ops().end(), [&](const Def* op) { return op->isa<Subst>(); });
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
        //rewrite(old_nom)->make_external();
        old_nom->unset();
        old_nom->make_internal();
    }

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    world().ILOG("finished");
    cleanup(world());
}

/*
a: [int, [bool, float]] -> b
b: [int, [bool, float]] -> (c#0, (c#1, c#2))
c: [int, bool, float]   -> (d#0, d#1, 23.f)
d: [int, bool]          -> (e#0, e#1)
e: [int, bool, int]

app A x
app B x
app C (x#0, x#1#0, x#1,1)
app D (x#0, x#1#0)
app E (x#0, x#1#0, y)
*/

/*
const Def* PassMan::rewrite(const Def* def) {
    if (def) {
        if (auto subst = def->isa<Subst>())
            return rewrite(subst->def(), subst->repls());
    }
    return def;
}

Def* PassMan::rewrite(Def* nom) {
    if (!has_subst(nom)) return nom;

    auto old_type  = nom->type();
    auto old_debug = nom->debug();
    auto old_ops   = nom->ops();

    auto new_type  = rewrite(old_type );
    auto new_debug = rewrite(old_debug);
    Array<const Def*> new_ops(old_ops, [&](const Def* def) { return rewrite(def); });

    for (auto op : nom->extended_ops()) {
        if (!analyze(op))
            return false;
    }
}
*/

uint32_t PassMan::rewrite(Def* cur_nom) {
    new_state(cur_nom);

    for (auto&& pass : passes_)
        pass->enter(cur_nom);

    auto undo = No_Undo;
    for (auto op : cur_nom->extended_ops()) {
        for (auto&& pass : passes_)
            undo = std::min(undo, pass->analyze(op));
    }

    while (undo != No_Undo && !cur_state().noms.empty()) {
        auto i = cur_state().noms.begin();
        auto next_nom = *i;
        rewrite(next_nom);
        cur_state().noms.erase(i);
    }

    states_.pop_back();
    return undo;
}

const Def* PassMan::rewrite(const Def* old_def, std::pair<const ReplArray, Def2Def>& repls) {
    if (old_def->is_const()) return old_def;

    if (auto old_param = old_def->isa<Param>()) {
        if (auto repl = repls.first.find(old_param))
            return repl->replacer;
    }

    // already rewritten in this or a prior state?
    for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
        if (auto new_def = repls.second.lookup(old_def))
            return *new_def;
    }

    /*
    if (auto subst = old_def->isa<Subst>()) {
        auto [it, succ] = local_.map.emplace(ReplArray(repls.first, subst->repls()), Def2Def());
        return rewrite(subst->def(), *it);
    }
    */

    auto new_type = rewrite(old_def->type(), repls);
    auto new_dbg  = old_def->debug() ? rewrite(old_def->debug(), repls) : nullptr;

    if (auto old_nom = old_def->isa_nominal()) {
        auto new_nom   = old_nom->stub(world(), new_type, new_dbg);

        for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i)
            new_nom->set(i, world().subst(old_nom->op(i), old_nom->param(), new_nom->param())); // TODO concat

        for (auto&& pass : passes_)
            pass->inspect(new_nom);

        return new_nom;
    } else {
        Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i), repls); });

        auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);
        for (auto&& pass : passes_)
            new_def = pass->rewrite(new_def);

        return repls.second[old_def] = new_def;
    }
}

uint32_t PassMan::analyze(const Def* def) {
    if (def->is_const()) return true;

    // already analyzed in this or a prior state?
    for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
        if (i->analyzed.contains(def)) return true;
    }
    // no? then, do it now
    cur_state().analyzed.emplace(def);

    if (auto nom = def->isa_nominal()) {
        cur_state().noms.emplace(nom);
        return true;
    }

    auto undo = No_Undo;
    for (auto op : def->extended_ops())
        undo = std::min(undo, analyze(op));

    world().DLOG("analyze: {}", def);
    for (auto&& pass : passes_)
        undo = std::min(undo, pass->analyze(def));

    return undo;
}

}
