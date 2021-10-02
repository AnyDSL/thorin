#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/util/container.h"

namespace thorin {

RWPassBase::RWPassBase(PassMan& man, const std::string& name)
    : man_(man)
    , name_(name)
    , proxy_id_(man.passes().size())
{}

FPPassBase::FPPassBase(PassMan& man, const std::string& name)
    : RWPassBase(man, name)
    , index_(man.fp_passes().size())
{}

void PassMan::push_state() {
    if (size_t num = fp_passes_.size()) {
        states_.emplace_back(num);

        // copy over from prev_state to curr_state
        auto&& prev_state      = states_[states_.size() - 2];
        curr_state().curr_nom  = prev_state.stack.top();
        curr_state().old_ops   = curr_state().curr_nom->ops();
        curr_state().stack     = prev_state.stack;
        curr_state().nom2visit = prev_state.nom2visit;

        for (size_t i = 0; i != num; ++i)
            curr_state().data[i] = fp_passes_[i]->copy(prev_state.data[i]);
    }
}

void PassMan::pop_states(size_t undo) {
    while (states_.size() != undo) {
        for (size_t i = 0, e = curr_state().data.size(); i != e; ++i)
            fp_passes_[i]->dealloc(curr_state().data[i]);

        if (undo != 0) // only reset if not final cleanup
            curr_state().curr_nom->set(curr_state().old_ops);

        states_.pop_back();
    }
}

void PassMan::run() {
    world().ILOG("run");

    auto num = fp_passes_.size();
    states_.emplace_back(num);
    for (size_t i = 0; i != num; ++i)
        curr_state().data[i] = fp_passes_[i]->alloc();

    for (auto pass : passes_)
        world().ILOG(" + {}", pass->name());
    world().debug_stream();

    auto externals = std::vector(world().externals().begin(), world().externals().end());
    for (const auto& [_, nom] : externals) {
        analyzed(nom);
        curr_state().stack.push(nom);
    }

    while (!curr_state().stack.empty()) {
        push_state();
        curr_nom_ = pop(curr_state().stack);
        world().VLOG("=== state {}: {} ===", states_.size() - 1, curr_nom_);

        if (!curr_nom_->is_set()) continue;

        for (auto pass : passes_) {
            if (pass->inspect()) pass->enter();
        }

        for (size_t i = 0, e = curr_nom_->num_ops(); i != e; ++i)
            curr_nom_->set(i, rewrite(curr_nom_->op(i)));

        world().VLOG("=== analyze ===");
        proxy_ = false;
        auto undo = No_Undo;
        for (auto op : curr_nom_->extended_ops())
            undo = std::min(undo, analyze(op));

        if (undo == No_Undo) {
            assert(!proxy_ && "proxies must not occur anymore after leaving a nom with No_Undo");
            world().DLOG("=== done ===");
        } else {
            pop_states(undo);
            world().DLOG("=== undo: {} -> {} ===", undo, curr_state().stack.top());
        }
    }

    world().ILOG("finished");
    pop_states(0);

    world().debug_stream();
    cleanup(world());
}

const Def* PassMan::rewrite(const Def* old_def) {
    if (old_def->no_dep()) return old_def;

    if (auto nom = old_def->isa_nom()) {
        curr_state().nom2visit.emplace(nom, curr_undo());
        return map(nom, nom);
    }

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

    if (auto proxy = new_def->isa<Proxy>()) {
        if (auto pass = static_cast<FPPassBase*>(passes_[proxy->id()]); pass->inspect()) {
            if (auto rw = pass->rewrite(proxy); rw != proxy)
                return map(old_def, rewrite(rw));
        }
    } else {
        for (auto pass : passes_) {
            if (!pass->inspect()) continue;

            if (auto var = new_def->isa<Var>()) {
                if (auto rw = pass->rewrite(var); rw != var)
                    return map(old_def, rewrite(rw));
            } else {
                if (auto rw = pass->rewrite(new_def); rw != new_def)
                    return map(old_def, rewrite(rw));
            }
        }
    }

    return map(old_def, new_def);
}

undo_t PassMan::analyze(const Def* def) {
    undo_t undo = No_Undo;

    if (def->no_dep() || analyzed(def)) {
        // do nothing
    } else if (auto nom = def->isa_nom()) {
        curr_state().stack.push(nom);
    } else if (auto proxy = def->isa<Proxy>()) {
        proxy_ = true;
        undo = static_cast<FPPassBase*>(passes_[proxy->id()])->analyze(proxy);
    } else {
        auto var = def->isa<Var>();

        if (!var) {
            for (auto op : def->extended_ops())
                undo = std::min(undo, analyze(op));
        }

        for (auto&& pass : fp_passes_) {
            if (pass->inspect())
                undo = std::min(undo, var ? pass->analyze(var) : pass->analyze(def));
        }
    }

    return undo;
}

}
