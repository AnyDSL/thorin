#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

void PassMan::push_state() {
    states_.emplace_back(num_passes());

    for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
        cur_state().data[i] = passes_[i]->alloc();

    if (states_.size() > 1) {
        auto&& prev_state   = states_[states_.size() - 2];
        cur_state().stack   = prev_state.stack; // copy over stack
        cur_state().cur_nom = prev_state.stack.top();
        cur_state().old_ops = cur_state().cur_nom->ops();
    }
}

void PassMan::pop_states(size_t undo) {
    while (states_.size() != undo) {
        for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
            passes_[i]->dealloc(cur_state().data[i]);

        if (undo != 0) // only reset if not final cleanup
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
        enqueued(nom);
        cur_state().stack.push(nom);
    }

    while (!cur_state().stack.empty()) {
        push_state();
        auto cur_nom = pop(cur_state().stack);
        world().DLOG("state/cur_nom: {}/{}", states_.size() - 1, cur_nom);

        if (!cur_nom->is_set()) continue;

        for (auto&& pass : passes_)
            pass->enter(cur_nom);

        for (size_t i = 0, e = cur_nom->num_ops(); i != e; ++i)
            cur_nom->set(i, rewrite(cur_nom, cur_nom->op(i)));

        for (auto&& pass : passes_)
            pass->finish(cur_nom);

        undo_t undo = No_Undo;
        for (auto&& pass : passes_)
            undo = std::min(undo, pass->analyze(cur_nom));

        if (undo == No_Undo) {
            for (auto op : cur_nom->extended_ops())
                enqueue(op);
        } else {
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
    for (auto&& pass : passes_)
        old_def = pass->prewrite(cur_nom, old_def);

    if (old_def->is_const() || old_def->isa<Proxy>()) return old_def;

    if (auto subst = old_def->isa<Subst>()) {
        map(subst->replacee(), subst->replacer());
        old_def = subst->def();
    }

    if (auto new_def = lookup(old_def)) return *new_def;
    if (auto nom = old_def->isa_nominal()) return map(nom, nom);

    auto new_type = rewrite(cur_nom, old_def->type());
    auto new_dbg  = old_def->debug() ? rewrite(cur_nom, old_def->debug()) : nullptr;

    Array<const Def*> new_ops(old_def->num_ops());
    for (size_t i = 0, e = old_def->num_ops(); i != e; ++i) {
        auto new_def = rewrite(cur_nom, old_def->op(i));
        new_ops[i] = new_def;
    }

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

    for (auto&& pass : passes_) {
        auto prev_def = new_def;
        new_def =  pass->rewrite(cur_nom, new_def);
        if (prev_def != new_def) new_def = rewrite(cur_nom, new_def);
    }

    return map(old_def, new_def);
}

void PassMan::enqueue(const Def* def) {
    if (def->is_const() || enqueued(def)) return;

    if (auto nom = def->isa_nominal()) {
        cur_state().stack.push(nom);
    } else {
        for (auto op : def->extended_ops())
            enqueue(op);
    }
}

}
