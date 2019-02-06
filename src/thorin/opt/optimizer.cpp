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
        rewrite(old_lam);
}

const Def* Optimizer::rewrite(const Def* old_def) {
    if (auto new_def = def2def_.lookup(old_def)) return *new_def;

    Lam* new_lam = nullptr;
    if (auto old_lam = old_def->isa_lam()) {
        new_lam = old_lam->stub(world(), old_lam->type());
        if (old_lam->is_external())
            new_lam->make_external();
        def2def_[old_lam] = new_lam;
    }

    Array<const Def*> ops(old_def->num_ops(), [&](size_t i) { return rewrite(old_def->op(i)); });

    if (new_lam) {
        new_lam->set_filter(ops[0]);
        new_lam->set_body(ops[1]);
        return new_lam;
    }

    auto new_def = old_def->rebuild(world(), old_def->type(), ops);
    for (auto&& opt : opts_)
        new_def = opt->visit(new_def);

    return def2def_[old_def] = new_def;
}

Optimizer std_optimizer(World& world) {
    Optimizer result(world);
    result.create<Inliner>();
    return result;
}

}
