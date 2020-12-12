#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

void PassMan::init_state() {
    auto num = fp_passes_.size();
    states_.emplace_back(num);

    for (size_t i = 0; i != num; ++i)
        cur_state().data[i] = fp_passes_[i]->alloc();
}

void PassMan::push_state() {
    if (!fp_passes_.empty()) {
        init_state();
        auto&& prev_state   = states_[states_.size() - 2];
        cur_state().stack   = prev_state.stack; // copy over stack
        cur_state().cur_nom = prev_state.stack.top();
        cur_state().old_ops = cur_state().cur_nom->ops();
    }
}

void PassMan::pop_states(size_t undo) {
    while (states_.size() != undo) {
        for (size_t i = 0, e = cur_state().data.size(); i != e; ++i)
            fp_passes_[i]->dealloc(cur_state().data[i]);

        if (undo != 0) // only reset if not final cleanup
            cur_state().cur_nom->set(cur_state().old_ops);

        states_.pop_back();
    }
}

void PassMan::run() {
    world().ILOG("run");
    init_state();

    for (auto pass : passes_)
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
        world().VLOG("=== state/cur_nom {}/{} ===", states_.size() - 1, cur_nom);

        if (!cur_nom->is_set()) continue;

        for (auto pass : passes_)
            pass->enter(cur_nom);

        for (size_t i = 0, e = cur_nom->num_ops(); i != e; ++i)
            cur_nom->set(i, rewrite(cur_nom, cur_nom->op(i)));

        for (auto pass : passes_)
            pass->finish(cur_nom);

        undo_t undo = No_Undo;
        for (auto&& pass : fp_passes_)
            undo = std::min(undo, pass->analyze(cur_nom));

        if (undo == No_Undo) {
            for (auto op : cur_nom->extended_ops())
                enqueue(op);
            world().DLOG("=== done ===");
        } else {
            pop_states(undo);
            world().DLOG("=== undo: {} -> {} ===", undo, cur_state().stack.top());
        }
    }

    world().ILOG("finished");
    pop_states(0);

    world().debug_stream();
    cleanup(world());
}

const Def* PassMan::rewrite(Def* cur_nom, const Def* old_def) {
    if (old_def->is_const()) return old_def;

    if (auto new_def = lookup(old_def)) {
        if (old_def == *new_def)
            return old_def;
        else
            return map(old_def, rewrite(cur_nom, *new_def));
    }

    auto new_type = rewrite(cur_nom, old_def->type());
    auto new_dbg  = old_def->dbg() ? rewrite(cur_nom, old_def->dbg()) : nullptr;

    // rewrite nominal
    if (auto old_nom = old_def->isa_nominal()) {
        for (auto pass : passes_) {
            if (auto rw = pass->rewrite(cur_nom, old_nom, new_type, new_dbg))
                return map(old_nom, rewrite(cur_nom, rw));
        }

        assert(old_nom->type() == new_type);
        return map(old_nom, old_nom);
    }

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(cur_nom, old_def->op(i)); });

    // rewrite structural before rebuild
    for (auto pass : passes_) {
        if (auto rw = pass->rewrite(cur_nom, old_def, new_type, new_ops, new_dbg))
            return map(old_def, rw);
    }

    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

    // rewrite structural after rebuild
    for (auto pass : passes_) {
        if (auto rw = pass->rewrite(cur_nom, new_def); rw != new_def)
            return map(old_def, rewrite(cur_nom, rw));
    }

    return map(old_def, new_def);
}

void PassMan::enqueue(const Def* def) {
    if (def->is_const() || enqueued(def)) return;
    assert(!def->isa<Proxy>() && "proxies must not occur anymore after finishing a nominal with No_Undo");

    if (auto nom = def->isa_nominal()) {
        cur_state().stack.push(nom);
    } else {
        for (auto op : def->extended_ops())
            enqueue(op);
    }
}

}
