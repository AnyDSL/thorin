#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

RWPass::RWPass(PassMan& man, const std::string& name)
    : man_(man)
    , name_(name)
    , proxy_id_(man.passes().size())
{}

FPPassBase::FPPassBase(PassMan& man, const std::string& name)
    : RWPass(man, name)
    , index_(man.fp_passes().size())
{}

void PassMan::push_state() {
    if (size_t num = fp_passes_.size()) {
        states_.emplace_back(num);

        auto&& prev_state   = states_[states_.size() - 2];
        cur_state().stack   = prev_state.stack; // copy over stack
        cur_state().cur_nom = prev_state.stack.top();
        cur_state().old_ops = cur_state().cur_nom->ops();

        for (size_t i = 0; i != num; ++i)
            cur_state().data[i] = fp_passes_[i]->copy(prev_state.data[i]);
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

    auto num = fp_passes_.size();
    states_.emplace_back(num);
    for (size_t i = 0; i != num; ++i)
        cur_state().data[i] = fp_passes_[i]->alloc();

    for (auto pass : passes_)
        world().ILOG(" + {}", pass->name());
    world().debug_stream();

    auto externals = std::vector(world().externals().begin(), world().externals().end());
    for (const auto& [_, nom] : externals) {
        analyzed(nom);
        cur_state().stack.push(nom);
    }

    while (!cur_state().stack.empty()) {
        push_state();
        cur_nom_ = pop(cur_state().stack);
        world().VLOG("=== state {}: {} ===", states_.size() - 1, cur_nom());

        if (!cur_nom()->is_set()) continue;

        for (auto pass : passes_) pass->enter();

        for (size_t i = 0, e = cur_nom()->num_ops(); i != e; ++i)
            cur_nom()->set(i, rewrite(cur_nom()->op(i)));

        for (auto pass : passes_) pass->leave();

        world().VLOG("=== analyze ===");
        proxy_ = false;
        auto undo = No_Undo;
        for (auto op : cur_nom()->extended_ops())
            undo = std::min(undo, analyze(op));

        if (undo == No_Undo) {
            assert(!proxy_ && "proxies must not occur anymore after leaving a nom with No_Undo");
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

const Def* PassMan::rewrite(const Def* old_def) {
    if (old_def->no_dep()) return old_def;
    if (auto old_nom = old_def->isa_nom()) return map(old_nom, old_nom);

    if (auto new_def = lookup(old_def)) {
        if (old_def == *new_def)
            return old_def;
        else
            return map(old_def, rewrite(*new_def));
    }

    auto new_type = rewrite(old_def->type());
    auto new_dbg  = old_def->dbg() ? rewrite(old_def->dbg()) : nullptr;

    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i)); });
    auto new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg);

    for (auto pass : passes_) {
        if (auto rw = pass->rewrite(new_def); rw != new_def)
            return map(old_def, rewrite(rw));
    }

    return map(old_def, new_def);
}

undo_t PassMan::analyze(const Def* def) {
    undo_t undo = No_Undo;

    if (def->no_dep() || analyzed(def)) {
        // do nothing
    } else if (auto nom = def->isa_nom()) {
        cur_state().stack.push(nom);
    } else if (auto proxy = def->isa<Proxy>()) {
        proxy_ = true;
        undo = static_cast<FPPassBase*>(passes_[proxy->id()])->analyze(proxy);
    } else {
        for (auto op : def->extended_ops())
            undo = std::min(undo, analyze(op));

        for (auto&& pass : fp_passes_)
            undo = std::min(undo, pass->analyze(def));
    }

    return undo;
}

}
