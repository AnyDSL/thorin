#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/util/log.h"

namespace thorin {

// TODO remove
template<typename T> void print_queue(T q) {
    std::cout << "  ";
    while(!q.empty()) {
        outf("{}/{} ", std::get<0>(q.top()), std::get<1>(q.top()));
        q.pop();
    }
    std::cout << '\n';
}

void PassMan::run() {
    states_.emplace_back(passes_);
    std::vector<const Def*> new_ops;

    auto externals = world().externals();
    for (auto lam : externals) {
        analyze(lam);

        while (!queue().empty()) {
            cur_nominal_ = std::get<Def*>(queue().top());
            outf("\ncur: {} {}\n", cur_state_id(), cur_nominal());
            outf("Q: ");
            print_queue(queue());

            bool mismatch = false;
            new_ops.resize(cur_nominal()->num_ops());
            for (size_t i = 0, e = cur_nominal()->num_ops(); i != e; ++i) {
                auto new_op = rewrite(cur_nominal()->op(i));
                mismatch |= new_op != cur_nominal()->op(i);
                new_ops[i] = new_op;
            }

            if (mismatch) {
                assert(undo_ == No_Undo && "only provoke undos during analyze");
                cur_nominal()->set(new_ops);
                continue;
            }

            queue().pop();
            for (auto op : cur_nominal()->ops())
                analyze(op);

            if (undo_ != No_Undo) {
                outf("undo: {} -> {}\n", cur_state_id(), undo_);

                assert(undo_ < cur_state_id());
                for (size_t i = cur_state_id(); i-- != undo_;)
                    states_[i].nominal->set(states_[i].old_ops);

                states_.resize(undo_);
                undo_ = No_Undo;
            }
        }
    }

    cleanup(world_);
}

Def* PassMan::rewrite(Def* old_nom) {
    assert(!lookup(old_nom).has_value());

    auto new_nom = old_nom;
    for (auto& pass : passes_)
        new_nom = pass->rewrite(new_nom);

    return map(old_nom, new_nom);
}

const Def* PassMan::rewrite(const Def* old_def) {
    if (auto new_def = lookup(old_def)) return *new_def;
    if (auto old_nom = old_def->isa_nominal()) return rewrite(old_nom);

    auto new_type = rewrite(old_def->type());

    bool changed = false;
    Array<const Def*> new_ops(old_def->num_ops(), [&](auto i) {
        auto new_op = rewrite(old_def->op(i));
        changed |= old_def->op(i) != new_op;
        return new_op;
    });

    auto new_def = changed ? old_def->rebuild(world(), new_type, new_ops) : old_def;
    for (auto& pass : passes_)
        new_def = pass->rewrite(new_def);

    return map(old_def, new_def);
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
