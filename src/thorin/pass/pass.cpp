#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
//#include "thorin/analyses/depends.h"

namespace thorin {

static bool has_subst(Def* nom) {
    return std::any_of(nom->extended_ops().begin(), nom->extended_ops().end(), [&](const Def* op) { return op->isa<Subst>(); });
}

bool PassMan::depends(Def* nom, Repls repls) const {
    return std::any_of(repls.begin(), repls.end(), [&](auto repl) { return deptree_.depends(nom, repl.replacee->nominal()); });
}

void PassMan::push_state() {
    states_.emplace_back(num_passes());
    for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
        cur_state().data[i] = passes_[i]->alloc();
}

void PassMan::pop_state() {
    for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
        passes_[i]->dealloc(cur_state().data[i]);
    states_.pop_back();
}

void PassMan::run() {
    world().ILOG("run");
    push_state();

    for (auto&& pass : passes_)
        world().ILOG(" + {}", pass->name());

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    auto& map = cur_state().map.emplace(ReplArray(), Def2Def()).first->second;
    auto externals = world().externals(); // copy
    for (const auto& [_, old_nom] : externals) {
        assert(old_nom->is_set() && "external must not be empty");

        auto new_nom = old_nom->stub(world(), old_nom->type(), old_nom->debug());
        for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i)
            new_nom->set(i, world().subst(old_nom->op(i), old_nom->param(), new_nom->param(), old_nom->op(i)->debug()));

        old_nom->unset();
        old_nom->make_internal();
        new_nom->make_external();

        map[old_nom] = new_nom;
    }

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    world().ILOG("finished");
    pop_state();
    assert(states_.empty());
    cleanup(world());
}

size_t PassMan::rewrite(Def* cur_nom) {
    if (!has_subst(cur_nom)) return No_Undo;

    size_t undo;
    do {
        push_state();

        for (auto&& pass : passes_)
            pass->enter(cur_nom);

        Array<const Def*> old_ops(cur_nom->ops());

        for (size_t i = 0, e = cur_nom->num_ops(); i != e; ++i) {
            if (auto subst = cur_nom->op(i)->isa<Subst>()) {
                auto [it, _] = cur_state().map.emplace(subst->repls(), Def2Def());
                cur_nom->set(i, rewrite(cur_nom, subst->def(), *it));
            }
        }

        undo = No_Undo;
        for (auto op : cur_nom->extended_ops()) {
            for (auto&& pass : passes_)
                undo = std::min(undo, pass->analyze(cur_nom, op));
        }

        while (undo != No_Undo && !cur_state().succs.empty()) {
            auto i = cur_state().succs.begin();
            undo = rewrite(*i);
            cur_state().succs.erase(i);
        }

        if (undo != No_Undo) cur_nom->set(old_ops);

        pop_state();
    } while (undo == cur_state_id());

    return undo;
}

const Def* PassMan::rewrite(Def* cur_nom, const Def* old_def, std::pair<const ReplArray, Def2Def>& pair) {
    auto& [repls, map] = pair;

    if (old_def->is_const()) return old_def;

    if (auto old_param = old_def->isa<Param>()) {
        if (auto repl = repls.find(old_param)) return repl->replacer;
    }

    // already rewritten in this or a prior state?
    if (auto new_def = map.lookup(old_def)) return *new_def;

    for (auto i = states_.rbegin() + 1, e = states_.rend(); i != e; ++i) {
        auto& state = *i;
        if (auto i = state.map.find(repls); i != state.map.end()) {
            if (auto new_def = i->second.lookup(old_def)) return *new_def;
        }
    }

    if (auto subst = old_def->isa<Subst>()) {
        auto [it, _] = cur_state().map.emplace(ReplArray(subst->repls(), repls), Def2Def());
        return rewrite(cur_nom, subst->def(), *it);
    }

    auto new_type = rewrite(cur_nom, old_def->type(), pair);
    auto new_dbg  = old_def->debug() ? rewrite(cur_nom, old_def->debug(), pair) : nullptr;

    const Def* new_def;
    if (auto old_nom = old_def->isa_nominal()) {
        if (depends(old_nom, repls)) {
            auto new_nom = old_nom->stub(world(), new_type, new_dbg);

            for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i)
                new_nom->set(i, world().subst(old_nom->op(i), old_nom->param(), new_nom->param(), repls, new_nom->op(i)->debug()));

            for (auto&& pass : passes_)
                pass->inspect(cur_nom, new_nom);

            new_def = new_nom;
        } else {
            new_def = old_nom;
        }
    } else {
        Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(cur_nom, old_def->op(i), pair); });
        auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

        for (auto&& pass : passes_)
            new_def = pass->rewrite(cur_nom, new_def);
    }

    return map[old_def] = new_def;
}

size_t PassMan::analyze(Def* cur_nom, const Def* def) {
    if (def->is_const()) return No_Undo;

    // already analyzed in this or a prior state?
    for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
        if (i->analyzed.contains(def)) return No_Undo;
    }
    // no? then, do it now
    cur_state().analyzed.emplace(def);

    if (auto nom = def->isa_nominal()) {
        cur_state().succs.emplace(nom);
        return No_Undo;
    }

    auto undo = No_Undo;
    for (auto op : def->extended_ops())
        undo = std::min(undo, analyze(cur_nom, op));

    world().DLOG("analyze: {}", def);
    for (auto&& pass : passes_)
        undo = std::min(undo, pass->analyze(cur_nom, def));

    return undo;
}

}
