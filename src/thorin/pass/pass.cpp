#include "thorin/pass/pass.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/importer.h"
#include "thorin/pass/inliner.h"

namespace thorin {

void PassMgr::run() {
    for (auto lam : world().externals())
        enqueue(lam);

    while (!cur_state().nominals.empty()) {
        auto def = pop(cur_state().nominals);

        bool mismatch = false;
        auto old_ops = def->ops();
        auto new_ops = Array<const Def*>(old_ops.size(), [&](auto i) {
            auto new_op = rewrite(old_ops[i]);
            mismatch |= new_op != old_ops[i];
            return new_op;
        });

        if (undo_ != No_Undo) {
            assert(undo_ > num_states());
            // undo required so roll back nominals to former ops
            for (size_t i = num_states(); i-- != undo_;) {
                for (auto&& [nom, ops] : states_[i].old_ops) {
                    nom->dump();
                    nom->as_nominal()->set(ops);
                }
            }

            states_.resize(undo_);
            undo_ = No_Undo;
        } else if (mismatch) {
            new_state();
            // memoize former ops in new state
            cur_state().old_ops.emplace(def, old_ops);
            def->set(new_ops);
        }

        for (auto op : new_ops)
            analyze(op);
    }

    // TODO provide this as stand alone method
    // get rid of garbage
    Importer importer(world_);
    importer.old2new_.rehash(world_.defs().capacity());

    for (auto external : world().externals())
        importer.import(external);

    swap(importer.world(), world_);
}

Def* PassMgr::rewrite(Def* old_nom) {
    assert(!lookup(old_nom).has_value());

    auto new_nom = old_nom;
    for (auto&& pass : passes_)
        new_nom = pass->rewrite(new_nom);

    return map(old_nom, new_nom);
}

const Def* PassMgr::rewrite(const Def* old_def) {
    if (auto new_def = lookup(old_def)) return *new_def;
    if (auto old_nom = old_def->isa_nominal()) return rewrite(old_nom);

    auto new_type = rewrite(old_def->type());

    bool rebuild = false;
    Array<const Def*> new_ops(old_def->num_ops(), [&](auto i) {
        auto old_op = old_def->op(i);
        auto new_op = rewrite(old_op);
        rebuild |= old_op != new_op;
        return new_op;
    });

    // only rebuild if necessary
    // this is not only an optimization but also required because some structural defs are not hash-consed
    auto new_def = rebuild ? old_def->rebuild(world(), new_type, new_ops) : old_def;

    for (auto&& pass : passes_)
        new_def = pass->rewrite(new_def);

    return map(old_def, new_def);
}

void PassMgr::analyze(const Def* def) {
    if (!cur_state().analyzed.emplace(def).second) return;
    if (auto nominal = def->isa_nominal()) return enqueue(nominal);

    for (auto op : def->ops())
        analyze(op);

    for (auto&& pass : passes_)
        pass->analyze(def);
}

}
