#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

void PassMan::push_state() {
    states_.emplace_back(num_passes());

    for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
        cur_state().data[i] = passes_[i]->alloc();

    if (states_.size() > 1) {
        const auto& prev_state = states_[states_.size() - 2];
        cur_state().stack = prev_state.stack; // copy over stack
        cur_state().cur_nom = prev_state.stack.top();
        cur_state().old_ops = cur_state().cur_nom->ops();
    }
}

void PassMan::pop_states(size_t undo) {
    while (states_.size() != undo) {
        for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
            passes_[i]->dealloc(cur_state().data[i]);

        if (undo != 0)// only reset if not final cleanup
            cur_state().cur_nom->set(cur_state().old_ops);

        states_.pop_back();
    }
}

void PassMan::run() {
    world().ILOG("run");
    push_state();

    for (auto&& pass : passes_)
        world().ILOG(" + {}", pass->name());
    world().debug_stream();

    for (const auto& [_, nom] : world().externals()) {
        map(nom, nom);
        analyzed(nom);
        cur_state().stack.push(nom);
    }

    while (!cur_state().stack.empty()) {
        push_state();
        auto cur_nom = pop(cur_state().stack);
        world().DLOG("cur_nom: {}", cur_nom);

        for (auto&& pass : passes_)
            pass->enter(cur_nom);

        if (!cur_nom->is_set()) continue;

        for (size_t i = 0, e = cur_nom->num_ops(); i != e; ++i)
            cur_nom->set(i, rewrite(cur_nom, cur_nom->op(i)));

        auto undo = No_Undo;
        for (auto op : cur_nom->extended_ops())
            undo = std::min(undo, analyze(cur_nom, op));

        if (undo != No_Undo) {
            pop_states(undo);
            world().DLOG("undo: {} - {}", undo, cur_state().stack.top());
        }
    }

    world().ILOG("finished");
    pop_states(0);

    world().debug_stream();
    cleanup(world());
}

const Def* PassMan::rewrite(Def* cur_nom, const Def* old_def) {
    if (old_def->is_const()) return old_def;
    if (auto new_def = lookup(old_def)) return *new_def;

    auto new_type = rewrite(cur_nom, old_def->type());
    auto new_dbg  = old_def->debug() ? rewrite(cur_nom, old_def->debug()) : nullptr;

    if (auto nom = old_def->isa_nominal()) {
        for (auto&& pass : passes_)
            pass->inspect(cur_nom, nom);

        return map(nom, nom);
    }

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(cur_nom, old_def->op(i)); });
    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

    for (auto&& pass : passes_) {
        auto prev_def = new_def;
        new_def = pass->rewrite(cur_nom, new_def);
        if (prev_def != new_def) new_def = rewrite(cur_nom, new_def);
    }

    return map(old_def, new_def);
}

size_t PassMan::analyze(Def* cur_nom, const Def* def) {
    if (def->is_const() || analyzed(def)) return No_Undo;

    auto undo = No_Undo;
    if (auto nom = def->isa_nominal()) {
        cur_state().stack.push(nom);
    } else {
        for (auto op : def->extended_ops())
            undo = std::min(undo, analyze(cur_nom, op));

        for (auto&& pass : passes_)
            undo = std::min(undo, pass->analyze(cur_nom, def));
    }

    return undo;
}

}
