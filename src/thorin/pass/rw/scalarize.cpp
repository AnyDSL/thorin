#include "thorin/pass/rw/scalarize.h"

#include "thorin/tuple.h"
#include "thorin/rewrite.h"

namespace thorin {

bool Scalerize::should_expand(Lam* lam) {
    if (ignore(lam) || keep_.contains(lam))
        return false;
    auto pi = lam->type();
    auto rewrite = lam->num_doms() > 1
        && pi->is_cn() && !pi->isa_nom(); // no ugly dependent pis
    if (!rewrite)
        keep_.emplace(lam);
    return rewrite;
}

Lam* Scalerize::make_scalar(Lam *lam) {
    if (auto sca_lam = tup2sca_.lookup(lam))
        return *sca_lam;
    auto types = DefVec();
    auto arg_sz = std::vector<size_t>();
    for (size_t i = 0; i < lam->num_doms(); i++) {
        auto n = flatten(types, lam->dom(i), false);
        arg_sz.push_back(n);
    }
    auto pi = world().cn(world().sigma(types));
    auto sca_lam = lam->stub(world(), pi, world().dbg("sca_" + lam->name()));
    size_t n = 0;
    world().DLOG("SCA type {} ~> {}", lam->type(), pi);
    auto new_vars = world().tuple(Array<const Def*>(lam->num_doms(), [&](auto i) {
        auto new_args = Array<const Def*>(arg_sz.at(i), [&](auto j) {
                return sca_lam->var(n + j);
        });
        n += arg_sz.at(i);
        return unflatten(new_args, lam->dom(i));
    }));
    sca_lam->set(lam->apply(new_vars));
    keep_.emplace(sca_lam);
    tup2sca_.emplace(lam, sca_lam);
    return sca_lam;
}

const Def* Scalerize::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        auto tup_lam = app->callee()->isa_nom<Lam>();

        if (!should_expand(tup_lam)) {
            return app;
        }

        auto sca_lam = make_scalar(tup_lam);

        world().DLOG("SCAL: lambda {} : {} ~> {} : {}", tup_lam, tup_lam->type(), sca_lam, sca_lam->type());
        auto new_args = DefVec();
        flatten(new_args, app->arg(), false);

        return world().app(sca_lam, new_args);
    }
    return def;
}

}
