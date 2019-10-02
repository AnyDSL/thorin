#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"

namespace thorin {

void PassMan::run() {
    states_.emplace_back(passes_);
    std::vector<const Def*> new_ops;

    auto externals = world().externals(); // copy
    for (const auto& [name, nom] : externals) {
        cur_nominal_ = nom;
        rewrite(nom); // provokes inspect
        analyze(nom); // puts into the queue
        cur_nominal_ = nullptr; // ensure to provoke pass->enter

        while (!queue().empty()) {
            auto old_nom = cur_nominal_;
            cur_nominal_ = std::get<Def*>(queue().top());

            if (old_nom != cur_nominal_) {
                new_state();
                for (auto& pass : passes_)
                    pass->enter(cur_nominal_);
            }

            world().DLOG("cur: {} {}", cur_state_id(), cur_nominal());

            bool mismatch = false;
            if (cur_nominal_->is_set()) {
                new_ops.resize(cur_nominal()->num_ops());
                for (size_t i = 0, e = cur_nominal()->num_ops(); i != e; ++i) {
                    auto new_op = rewrite(cur_nominal()->op(i));
                    mismatch |= new_op != cur_nominal()->op(i);
                    new_ops[i] = new_op;
                }
            }

            if (mismatch) {
                assert(undo_ == No_Undo && "only provoke undos during analyze");
                cur_nominal()->set(new_ops);
                continue;
            }

            queue().pop();

            if (cur_nominal_->is_set()) {
                for (auto op : cur_nominal()->ops())
                    analyze(op);
            }

            if (undo_ != No_Undo) {
                world().DLOG("undo: {} -> {}", cur_state_id(), undo_);

                for (size_t i = cur_state_id(); i-- != undo_;) {
                    if (states_[i].nominal->is_set())
                        states_[i].nominal->set(states_[i].old_ops);
                }

                states_.resize(undo_);
                undo_ = No_Undo;
                cur_nominal_ = std::get<Def*>(queue().top()); // don't provoke pass->enter
            }
        }
    }

    cleanup(world_);
}

const Def* PassMan::rewrite(const Def* old_def) {
    if (auto new_def = lookup(old_def)) return *new_def;

    if (auto nominal = old_def->isa_nominal()) {
        for (auto& pass : passes_)
            pass->inspect(nominal);
        return map(nominal, nominal);
    }

    auto new_type  = rewrite(old_def->type());

    bool changed = false;
    Array<const Def*> new_ops(old_def->num_ops(), [&](auto i) {
        auto new_op = rewrite(old_def->op(i));
        changed |= old_def->op(i) != new_op;
        return new_op;
    });

    auto new_def = changed ? old_def->rebuild(world(), new_type, new_ops, old_def->debug()) : old_def;

    for (auto& pass : passes_)
        new_def = pass->rewrite(new_def);

    assert(!cur_state().old2new.contains(new_def) || cur_state().old2new[new_def] == new_def);
    return map(old_def, map(new_def, new_def));
}

void PassMan::analyze(const Def* def) {
    if (!cur_state().analyzed.emplace(def).second) return;
    if (auto nominal = def->isa_nominal()) return queue().emplace(nominal, time_++);

    for (auto op : def->ops())
        analyze(op);

    for (auto& pass : passes_)
        pass->analyze(def);
}

}
