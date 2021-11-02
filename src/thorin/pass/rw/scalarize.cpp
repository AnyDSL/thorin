#include "thorin/pass/rw/scalarize.h"

#include "thorin/tuple.h"
#include "thorin/rewrite.h"

namespace thorin {

// TODO should also work for nominal non-dependent sigmas

// TODO merge with make_scalar
bool Scalerize::should_expand(Lam* lam) {
    if (ignore(lam)) return false;
    if (auto sca_lam = tup2sca_.lookup(lam); sca_lam && *sca_lam == lam) return false;

    auto pi = lam->type();
    if (lam->num_doms() > 1 && pi->is_cn() && !pi->isa_nom()) return true; // no ugly dependent pis

    tup2sca_[lam] = lam;
    return false;
}

Lam* Scalerize::make_scalar(Lam* tup_lam) {
    if (auto sca_lam = tup2sca_.lookup(tup_lam)) return *sca_lam;

    auto types = DefVec();
    auto arg_sz = std::vector<size_t>();
    bool todo = false;
    for (size_t i = 0, e = tup_lam->num_doms(); i != e; ++i) {
        auto n = flatten(types, tup_lam->dom(i), false);
        arg_sz.push_back(n);
        todo |= n != 1;
    }

    if (!todo) return tup2sca_[tup_lam] = tup_lam;

    auto pi = world().cn(world().sigma(types));
    auto sca_lam = tup_lam->stub(world(), pi, tup_lam->dbg());
    if (eta_exp_) eta_exp_->new2old(sca_lam, tup_lam);
    size_t n = 0;
    world().DLOG("type {} ~> {}", tup_lam->type(), pi);
    auto new_vars = world().tuple(Array<const Def*>(tup_lam->num_doms(), [&](auto i) {
        auto new_args = Array<const Def*>(arg_sz.at(i), [&](auto j) {
                return sca_lam->var(n + j);
        });
        n += arg_sz.at(i);
        return unflatten(new_args, tup_lam->dom(i));
    }));
    sca_lam->set(tup_lam->apply(new_vars));
    tup2sca_[sca_lam] = sca_lam;
    tup2sca_.emplace(tup_lam, sca_lam);

    return sca_lam;
}

const Def* Scalerize::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        auto tup_lam = app->callee()->isa_nom<Lam>();

        if (!should_expand(tup_lam)) return app;

        if (auto sca_lam = make_scalar(tup_lam); sca_lam != tup_lam) {
            world().DLOG("lambda {} : {} ~> {} : {}", tup_lam, tup_lam->type(), sca_lam, sca_lam->type());
            auto new_args = DefVec();
            flatten(new_args, app->arg(), false);

            return world().app(sca_lam, new_args);
        }
    }
    return def;
}

}
