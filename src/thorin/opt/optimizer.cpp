#include "thorin/opt/optimizer.h"

#include "thorin/analyses/scope.h"
#include "thorin/opt/inliner.h"

namespace thorin {

#if 0
void swap(Optimizer& a, Optimizer& b);
    using std::swap;
    swap(a.world_, b.world_);
    swap(a.opts_,  b.opts_);
}
#endif

void Optimizer::run() {
    auto externals = world().externals();

    for (auto old_lam : externals)
        enqueue(old_lam);

    while (!nominals_.empty()) {
        auto def = pop(nominals_);

        auto num_ops = def->num_ops();
        Array<const Def*> new_ops(num_ops, [&](size_t i) { return rewrite(def->op(i)); });

        for (size_t i = 0; i != num_ops; ++i)
            def->set(i, new_ops[i]);
    }

    for (auto old_lam : externals) {
        //old_lam->make_internal();
        lookup(old_lam)->as_nominal<Lam>()->make_external();
    }
}

void Optimizer::enqueue(Def* old_def) {
    if (old2new_.contains(old_def)) return;

    auto new_def = old_def;
    /* TODO
    for (auto&& opt : opts_)
        new_def = opt->visit(new_def);
    */
    old2new_[old_def] = new_def;
    nominals_.push(new_def);
}

const Def* Optimizer::rewrite(const Def* old_def) {
    if (auto nominal = old_def->isa_nominal()) enqueue(nominal);
    if (auto new_def = old2new_.lookup(old_def)) return *new_def;

    auto new_type = rewrite(old_def->type());

    bool rebuild = false;
    Array<const Def*> new_ops(old_def->num_ops(), [&](size_t i) {
        auto old_op = old_def->op(i);
        auto new_op = rewrite(old_op);
        rebuild |= old_op != new_op;
        return new_op;
    });

    auto new_def = rebuild ? old_def->rebuild(world(), new_type, new_ops) : old_def;

    for (auto&& opt : opts_)
        new_def = opt->rewrite(new_def);

    return old2new_[old_def] = new_def;
}

Optimizer std_optimizer(World& world) {
    Optimizer result(world);
    result.create<Inliner>();
    return result;
}

}
