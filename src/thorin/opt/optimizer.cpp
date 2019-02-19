#include "thorin/opt/optimizer.h"

#include "thorin/analyses/scope.h"
#include "thorin/opt/inliner.h"

namespace thorin {

void Optimizer::run() {
    auto externals = world().externals();

    for (auto lam : externals)
        nominals_.push(lam);

    while (!nominals_.empty()) {
        auto def = pop(nominals_);

        bool mismatch = false;
        auto old_ops = def->ops();
        auto new_ops = Array<const Def*>(old_ops.size(), [&](auto i) {
            auto new_op = rewrite(old_ops[i]);
            mismatch |= new_op != old_ops[i];
            return new_op;
        });

        if (mismatch) {
            // TODO install restore point
            def->set(new_ops);
        }

        for (auto op : new_ops)
            analyze(op);
    }
}

Def* Optimizer::rewrite(Def* old_nom) {
    auto new_nom = old_nom;
    for (auto&& opt : opts_)
        new_nom = opt->rewrite(new_nom);

    old2new_[old_nom] = new_nom;

    return new_nom;
}

const Def* Optimizer::rewrite(const Def* old_def) {
    if (auto new_def = old2new_.lookup(old_def)) return *new_def;
    if (auto old_nom = old_def->isa_nominal()) return rewrite(old_nom);

    auto new_type = rewrite(old_def->type());

    bool rebuild = false;
    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) {
        auto old_op = old_def->op(i);
        auto new_op = rewrite(old_op);
        rebuild |= old_op != new_op;
        return new_op;
    });

    // only rebuild if necessary
    // this is not only an optimization but also required because some structural defs are not hash-consed
    auto new_def = rebuild ? old_def->rebuild(world(), new_type, new_ops) : old_def;

    for (auto&& opt : opts_)
        new_def = opt->rewrite(new_def);

    return old2new_[old_def] = new_def;
}

void Optimizer::analyze(const Def* def) {
    if (!analyzed_.emplace(def).second) return;
    if (auto nominal = def->isa_nominal()) return nominals_.push(nominal);

    for (auto op : def->ops())
        analyze(op);

    for (auto&& opt : opts_)
        opt->analyze(def);
}

Optimizer std_optimizer(World& world) {
    Optimizer result(world);
    result.create<Inliner>();
    return result;
}

}
