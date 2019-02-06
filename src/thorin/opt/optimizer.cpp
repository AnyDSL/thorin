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
    old2new_[world().branch()]    = world().branch();
    old2new_[world().end_scope()] = world().end_scope();
    old2new_[world().universe()]  = world().universe();

    auto externals = world().externals();

    for (auto old_lam : externals)
        nominals_.push(old_lam);

    while (!nominals_.empty()) {
        //auto nominal = nominals_.pop();
    }

    for (auto old_lam : externals) {
        //old_lam->make_internal();
        lookup(old_lam)->as_nominal<Lam>()->make_external();
    }
}

const Def* Optimizer::rewrite(const Def* old_def) {
    if (auto new_def = old2new_.lookup(old_def)) return *new_def;

    auto new_type = rewrite(old_def->type());

    const Def* new_def = nullptr;
    if (old_def->is_nominal()) {
        new_def = old_def->stub(world(), new_type);
        old2new_[old_def] = new_def;
    }

    auto num_ops = old_def->num_ops();
    Array<const Def*> new_ops(num_ops, [&](size_t i) { return rewrite(old_def->op(i)); });

    if (new_def) {
        for (size_t i = 0; i != num_ops; ++i)
            const_cast<Def*>(new_def)->set(i, new_ops[i]);
    } else {
        new_def = old_def->rebuild(world(), new_type, new_ops);
        old2new_[old_def] = new_def;

        for (auto&& opt : opts_)
            new_def = opt->visit(new_def);
        //new_def = rewrite(new_def);
        old2new_[old_def] = new_def;
    }

    return new_def;
}

Optimizer std_optimizer(World& world) {
    Optimizer result(world);
    result.create<Inliner>();
    return result;
}

}
