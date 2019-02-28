#include "thorin/pass/pass.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/importer.h"
#include "thorin/util/log.h"

namespace thorin {

// TODO put this somewhere else
static void cleanup(World& world) {
    Importer importer(world);
    importer.old2new_.rehash(world.defs().capacity());

    for (auto external : world.externals())
        importer.import(external);

    swap(importer.world(), world);
}

// TODO remove
template<typename T> void print_queue(T q) {
    std::cout << "  ";
    while(!q.empty()) {
        outf("{}/{} ", std::get<0>(q.top()), std::get<1>(q.top()));
        q.pop();
    }
    std::cout << '\n';
}

World& PassBase::world() { return mgr().world(); }

void PassMgr::run() {
    states_.emplace_back(passes_);
    std::vector<const Def*> new_ops;

    auto externals = world().externals();
    for (auto lam : externals) {
        analyze(lam);

        while (!queue().empty()) {
            cur_nominal_ = std::get<Def*>(queue().top());

            auto old_id [[maybe_unused]] = state_id();
            outf("\ncur: {} {}\n", state_id(), cur_nominal_);
            outf("Q: ");
            print_queue(queue());

            bool mismatch = false;
            new_ops.resize(cur_nominal_->num_ops());
            for (size_t i = 0, e = cur_nominal_->num_ops(); i != e; ++i) {
                auto new_op = rewrite(cur_nominal_->op(i));
                mismatch |= new_op != cur_nominal_->op(i);
                new_ops[i] = new_op;
            }

            if (mismatch) {
                //assert(old_id < state_id() && "nominal changed; you must introduce a new state");
                assert(undo_ == No_Undo && "only provoke undos in the analyze phase");
                cur_nominal_->set(new_ops);
                continue;
            }

            analyze();

            while (undo_ != No_Undo) {
                outf("undo: {} -> {}\n", state_id(), undo_);

                assert(undo_ < state_id());
                for (size_t i = state_id(); i-- != undo_;)
                    states_[i].nominal->set(states_[i].old_ops);

                states_.resize(undo_);
                undo_ = No_Undo;
                analyze();
            }

            queue().pop();
        }
    }

    // TODO put this somewhere else
    cleanup(world_);
}

Def* PassMgr::rewrite(Def* old_nom) {
    assert(!lookup(old_nom).has_value());

    auto new_nom = old_nom;
    for (auto&& pass : passes_)
        new_nom = pass->rewrite(new_nom);

    return map(old_nom, new_nom);
}

const Def* PassMgr::rewrite(const Def* old_def) {
    auto new_def = rebuild(old_def);

    for (auto&& pass : passes_)
        new_def = pass->rewrite(new_def);

    //if (old_def != new_def)
        //outf("map: {} -> {}\n", old_def, new_def);

    return map(old_def, new_def);
}

const Def* PassMgr::rebuild(const Def* old_def) {
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
    return rebuild ? old_def->rebuild(world(), new_type, new_ops) : old_def;
}

void PassMgr::analyze() {
    for (auto op : std::get<Def*>(queue().top())->ops())
        analyze(op);
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
