#include "thorin/pass/pass.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/importer.h"
#include "thorin/util/log.h"

namespace thorin {

void PassMgr::run() {
    for (auto lam : world().externals())
        enqueue(lam);

    std::vector<const Def*> new_ops;

    while (!cur_state().nominals.empty()) {
        auto nominal = cur_state().nominals.top();

        bool mismatch = false;
        new_ops.resize(nominal->num_ops());
        for (size_t i = 0, e = nominal->num_ops(); i != e; ++i) {
            auto new_op = rewrite(nominal->op(i));
            mismatch |= new_op != nominal->op(i);
            new_ops[i] = new_op;
        }

        if (mismatch) {
            assert(undo_ == No_Undo && "only provoke undos in the analyze phase");
            new_state(nominal, nominal->ops());
            nominal->set(new_ops);
            continue;
        }

        for (auto op : new_ops)
            analyze(op);

        if (undo_ != No_Undo) {
            assert(undo_ < num_states());

            for (size_t i = num_states(); i-- != undo_;)
                states_[i].nominal->set(states_[i].old_ops);

            outf("undo: {} -> {}\n", num_states(), undo_);
            states_.resize(undo_);

            for (auto&& pass : passes_)
                pass->undo(undo_);

            undo_ = No_Undo;
        }

        cur_state().nominals.pop();
    }

    //world_.dump();

    // TODO provide this as stand alone method
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
