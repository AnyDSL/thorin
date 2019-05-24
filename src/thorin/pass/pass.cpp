#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/util/log.h"

namespace thorin {

void PassMan::run() {
    states_.emplace_back(passes_);
    std::vector<const Def*> new_ops;

    auto externals = world().externals();
    for (auto lam : externals) {
        cur_nominal_ = lam;
        rewrite(lam); // provokes inspect
        analyze(lam); // puts into the queue
        cur_nominal_ = nullptr; // ensure to provoke pass->enter

        while (!queue().empty()) {
            auto old_nom = cur_nominal_;
            cur_nominal_ = std::get<Def*>(queue().top());
            new_state(); // TODO do we need this?

            if (old_nom != cur_nominal_) {
                for (auto& pass : passes_)
                    pass->enter(cur_nominal_);
            }

            outf("\ncur: {} {}\n", cur_state_id(), cur_nominal());

            new_ops.resize(cur_nominal()->num_ops());
            for (size_t i = 0, e = cur_nominal()->num_ops(); i != e; ++i)
                new_ops[i] = rewrite(cur_nominal()->op(i));
            cur_nominal()->set(new_ops);

            queue().pop();
            for (auto op : cur_nominal()->ops())
                analyze(op);
        }
    }

    cleanup(world_);
}

bool PassMan::push(const Def* old_def) {
    if (lookup(old_def)) return false;

    if (auto nominal = old_def->isa_nominal()) {
        for (auto& pass : passes_)
            pass->inspect(nominal);
        map(nominal, nominal);
        return false;
    }

    stack().emplace(old_def);
    return true;
}

const Def* PassMan::rewrite(const Def* root) {
    std::vector<const Def*> new_ops;
    push(root);

    while (!stack().empty()) {
        auto old_def = stack().top();
        auto n = old_def->num_ops();
        new_ops.resize(n);

        bool todo = false;
        todo |= push(old_def->type());
        for (size_t i = 0; i != n; ++i)
            todo |= push(old_def->op(i));

        if (!todo) {
            auto new_type = *lookup(old_def->type());

            bool changed = new_type != old_def->type();
            for (size_t i = 0; i != n; ++i)
                changed |= old_def->op(i) != (new_ops[i] = *lookup(old_def->op(i)));

            auto new_def = changed ? old_def->rebuild(world(), new_type, new_ops) : old_def;

            for (auto& pass : passes_)
                new_def = pass->rewrite(new_def);

            //assert(!cur_state().old2new.contains(new_def) || cur_state().old2new[new_def] == new_def);
            map(old_def, map(new_def, new_def));

            stack().pop();
        }
    }

    return *lookup(root);
}

void PassMan::analyze(const Def* def) {
    if (!cur_state().analyzed.emplace(def).second) return;
    if (auto nominal = def->isa_nominal()) return queue().emplace(nominal, time_++);

    for (auto op : def->ops())
        analyze(op);

    for (auto& pass : passes_)
        pass->analyze(def);
}

void PassMan::undo(size_t u) {
    outf("undo: {} -> {}\n", cur_state_id(), u);

    for (size_t i = cur_state_id(); i --> u;)
        states_[i].nominal->set(states_[i].old_ops);

    states_.resize(u);
    cur_nominal_ = std::get<Def*>(queue().top()); // don't provoke pass->enter
}

}
