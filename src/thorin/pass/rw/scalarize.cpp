#include "thorin/pass/rw/scalarize.h"

#include "thorin/tuple.h"
#include "thorin/rewrite.h"
#include "thorin/pass/fp/eta_exp.h"

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

Lam* Scalerize::make_scalar(const Def* def) {
    auto tup_lam = def->isa_nom<Lam>();
    assert(tup_lam);

    if (auto sca_lam = tup2sca_.lookup(tup_lam)) 
        return *sca_lam;

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
    auto new_vars = world().tuple(DefArray(tup_lam->num_doms(), [&](auto i) {
        auto tuple = DefArray(arg_sz.at(i), [&](auto j) {
            return sca_lam->var(n++);
        });
        return unflatten(tuple, tup_lam->dom(i));
    }));
    sca_lam->set(tup_lam->apply(new_vars));
    tup2sca_[sca_lam] = sca_lam;
    tup2sca_.emplace(tup_lam, sca_lam);
    world().DLOG("lambda {} : {} ~> {} : {}", tup_lam, tup_lam->type(), sca_lam, sca_lam->type());
    return sca_lam;
}

const Def* Scalerize::rewrite(const Def* def) {
    auto& w = world();
    if (auto app = def->isa<App>()) {
        const Def* sca_callee = app->callee();
        // auto tup_lam = app->callee()->isa_nom<Lam>();

        // if (!should_expand(tup_lam)) return app;
        if (auto tup_lam = sca_callee->isa_nom<Lam>(); should_expand(tup_lam)) {
            sca_callee = make_scalar(tup_lam);

        } else if (auto proj = sca_callee->isa<Extract>()) {
            auto tuple = proj->tuple()->isa<Tuple>();
            if (tuple && std::all_of(tuple->ops().begin(), tuple->ops().end(), 
                    [&](const Def* op) { return should_expand(op->isa_nom<Lam>()); })) {
                auto new_tuple = w.tuple(DefArray(tuple->num_ops(), [&](auto i) { 
                    return make_scalar(tuple->op(i)); 
                }));
                sca_callee = w.extract(new_tuple, proj->index());
                w.DLOG("Expand tuple: {, } ~> {, }", tuple->ops(), new_tuple->ops());
            }
        }

        if (sca_callee != app->callee()) {
            auto new_args = DefVec();
            flatten(new_args, app->arg(), false);
            return world().app(sca_callee, new_args);
        }
    }
    return def;
}

}
