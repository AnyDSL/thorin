#include "thorin/pass/pass.h"

#include "thorin/rewrite.h"
#include "thorin/util/log.h"

namespace thorin {

void dump_stack(std::stack<const Def*> stack) {
    std::cout << "push: \n";
    while (!stack.empty()) {
        std::cout << " | ";
        pop(stack)->dump();
    }
    std::cout << "\n";
}


void PassMan::run() {
    states_.emplace_back(passes_);
    std::vector<const Def*> new_ops;

    auto externals = world().externals();
    for (auto lam : externals) {
        cur_nominal_ = lam;
        push(lam); // provokes inspect
        queue().emplace(lam, time_++);

        while (!queue().empty()) {
            cur_nominal_ = std::get<Def*>(queue().top());
            for (auto& pass : passes_)
                pass->enter(cur_nominal_);

            for (auto op : cur_nominal()->ops())
                push(op);

            for (bool todo = true; todo;) {
                outf("\ncur: {} {}\n", cur_state_id(), cur_nominal());

                todo = false;

                rewrite();

                outf("pop: {}\n", std::get<Def*>(queue().top()));
                queue().pop();

                for (auto op : cur_nominal()->ops()) {
                    if (auto undo = analyze(op, PassBase::No_Undo); undo != PassBase::No_Undo) {
                        todo = true;
                        outf("undo: {} -> {}\n", cur_state_id(), undo);

                        for (size_t i = cur_state_id(); i --> undo;) {
                            for (const auto& [nom, old_ops] : states_[i].nominals)
                                nom->set(old_ops);
                        }

                        states_.resize(undo);
                        cur_nominal_ = std::get<Def*>(queue().top());
                        break;
                    }
                }
            }
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

    std::cout << std::flush;
    assert(stack().empty() || stack().top() != old_def);

    stack().emplace(old_def);
    //dump_stack(stack());
    return true;
}

void PassMan::rewrite() {
    std::vector<const Def*> new_ops;

    while (!stack().empty()) {
        auto old_def = stack().top();
        auto n = old_def->num_ops();
        new_ops.resize(n);

        bool todo = false;
        todo |= push(old_def->type());
        for (size_t i = 0; i != n; ++i)
            todo |= push(old_def->op(i));

        if (!todo) {
            stack().pop();
            auto new_type = *lookup(old_def->type());

            bool changed = new_type != old_def->type();
            for (size_t i = 0; i != n; ++i)
                changed |= old_def->op(i) != (new_ops[i] = *lookup(old_def->op(i)));

            auto new_def = changed ? old_def->rebuild(world(), new_type, new_ops) : old_def;

            for (auto& pass : passes_)
                new_def = pass->rewrite(new_def);

            //assert(!cur_state().old2new.contains(new_def) || cur_state().old2new[new_def] == new_def);
            map(old_def, map(new_def, new_def));
        }
    }

    cur_state().nominals.emplace_back(cur_nominal(), cur_nominal()->ops());
    for (size_t i = 0, e = cur_nominal()->num_ops(); i != e; ++i)
        cur_nominal()->set(i, *lookup(cur_nominal()->op(i)));
}

size_t PassMan::analyze(const Def* def, size_t undo) {
    if (!cur_state().analyzed.emplace(def).second) return undo;
    if (auto nominal = def->isa_nominal()) {
        queue().emplace(nominal, time_++);
        return undo;
    }

    for (auto op : def->ops())
        undo = std::min(undo, analyze(op, undo));

    for (size_t i = 0, e = passes_.size(); i != e && undo == PassBase::No_Undo; ++i)
        undo = passes_[i]->analyze(def);

    return undo;
}

}
