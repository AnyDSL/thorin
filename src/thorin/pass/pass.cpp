#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

/*
 * helpers
 */

void PassMan::push_state() {
    states_.emplace_back(num_passes());
    for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
        cur_state().data[i] = passes_[i]->alloc();

    if (states_.size() > 1)
        cur_state().stack = states_[states_.size() - 2].stack;
}

void PassMan::pop_states(size_t undo) {
    while (states_.size() != undo) {
        for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
            passes_[i]->dealloc(cur_state().data[i]);
        states_.pop_back();
    }
}

std::optional<const Def*> PassMan::lookup(const Def* old_def) {
    for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
        const auto& old2new = i->old2new;
        if (auto i = old2new.find(old_def); i != old2new.end()) return i->second;
    }

    return {};
}

bool PassMan::analyzed(const Def* def) {
    for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
        if (i->analyzed.contains(def)) return true;
    }

    cur_state().analyzed.emplace(def);
    return false;
}

Def* PassMan::stub(Def* old_nom, const Def* type, const Def* dbg) {
    auto new_nom = old_nom->stub(world(), type, dbg);
    size_t i = 0;

    if (old_nom->is_set()) {
        for (auto op : old_nom->ops())
            new_nom->set(i++, world().subst(op, old_nom->param(), new_nom->param(), op->debug()));
    }

    return new_nom;
}

//------------------------------------------------------------------------------

void PassMan::run() {
    world().ILOG("run");
    push_state();

    for (auto&& pass : passes_)
        world().ILOG(" + {}", pass->name());

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());

    Array<std::array<Def*, 2>> old_new_exts(world().externals().size());
    size_t i = 0;
    for (const auto& [_, old_nom] : world().externals())
        old_new_exts[i++] = {old_nom, stub(old_nom, old_nom->type(), old_nom->debug())};

    for (const auto& [old_nom, new_nom] : old_new_exts) {
        old_nom->unset();
        old_nom->make_internal();
        new_nom->make_external();
        map(old_nom, new_nom);
        analyzed(new_nom);
        cur_state().stack.push(new_nom);
    }

    loop();

    world().ILOG("finished");
    pop_states(0);
    assert(states_.empty());
    cleanup(world());

    if (world().min_level() == LogLevel::Debug)
        world().stream(world().stream());
}

void PassMan::loop() {
    while (!cur_state().stack.empty()) {
        push_state();
        auto cur_nom = pop(cur_state().stack);
        if (!cur_nom->is_set()) continue;

        for (size_t i = 0, e = cur_nom->num_ops(); i != e; ++i) {
            if (auto subst = cur_nom->op(i)->isa<Subst>())
                cur_nom->set(i, rewrite(cur_nom, subst));
        }

        auto undo = No_Undo;
        for (auto op : cur_nom->extended_ops())
            undo = std::min(undo, analyze(cur_nom, op));

        if (undo != No_Undo) pop_states(undo);
    }
}

const Def* PassMan::rewrite(Def* cur_nom, const Def* old_def) {
    if (old_def->is_const()) return old_def;
    if (auto new_def = lookup(old_def)) return *new_def;

    if (auto subst = old_def->isa<Subst>()) {
        map(subst->replacee(), subst->replacer());
        return rewrite(cur_nom, subst->def());
    }

    auto new_type = rewrite(cur_nom, old_def->type());
    auto new_dbg  = old_def->debug() ? rewrite(cur_nom, old_def->debug()) : nullptr;

    if (auto old_nom = old_def->isa_nominal()) {
        auto new_nom = stub(old_nom, new_type, new_dbg);

        for (auto&& pass : passes_)
            pass->inspect(cur_nom, new_nom);

        return map(old_nom, new_nom);
    }

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(cur_nom, old_def->op(i)); });
    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

    for (auto&& pass : passes_)
        new_def = pass->rewrite(cur_nom, new_def);

    return map(old_def, new_def);
}

size_t PassMan::analyze(Def* cur_nom, const Def* def) {
    if (def->is_const() || analyzed(def)) return No_Undo;

    if (auto nom = def->isa_nominal()) {
        cur_state().stack.push(nom);
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
