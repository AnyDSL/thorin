
#include "thorin/tuple.h"
#include "thorin/rewrite.h"

#include "thorin/pass/fp/scalarize.h"

namespace thorin {

using DefVec = std::vector<const Def*>;

bool Scalerize::should_expand(Lam* lam) {
    auto uses = lam->uses();
    auto eta_expanded = std::all_of(uses.begin(), uses.end(), [&](auto use) {
        return isa_callee(use.def(), use.index());
    });
    auto pi = lam->type();
    return eta_expanded
        && pi->is_cn() && !pi->isa_nom() // no ugly dependet type
        && lam->body() && lam->filter(); // no imported def
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
    auto sca_lam = world().nom_lam(pi, world().dbg("sca_" + lam->name()));
    auto rewriter = Rewriter(world());
    for (int i = 0; i < lam->num_doms(); i++) {
        auto new_arg = 
            world().tuple(Array<const Def*>(arg_sz.at(i), [&](auto j) {
                return sca_lam->var(i + j);
            }));
        rewriter.old2new.emplace(lam->var(i), new_arg);
    }
    sca_lam->set_body(rewriter.rewrite(lam->body()));
    sca_lam->set_body(rewriter.rewrite(lam->filter()));
    keep_.emplace(sca_lam);
    tup2sca_.emplace(lam, sca_lam);
    return lam;
}


const Def* Scalerize::rewrite(const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto tup_lam = app->callee()->isa_nom<Lam>();
    if (ignore(tup_lam) || tup_lam->num_vars() <= 1 || keep_.contains(tup_lam)) return app;

    if (!should_expand(tup_lam)) {
        keep_.emplace(tup_lam);
        return app;
    }

    auto sca_lam = make_scalar(tup_lam);
    auto new_args = std::vector<const Def*>();
    flatten(new_args, app->arg(), false);

    return world().app(sca_lam, new_args);
}

}

#if 0
#include "thorin/pass/fp/scalarize.h"

namespace thorin {

const Def* Scalerize::rewrite(const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto tup_lam = app->callee()->isa_nom<Lam>();
    if (ignore(tup_lam) || tup_lam->num_vars() <= 1 || keep_.contains(tup_lam)) return app;

    auto& sca_lam = tup2sca_.emplace(tup_lam, nullptr).first->second;

    if (sca_lam == nullptr) {
        // auto&& [args, _, __] = insert<LamMap<Args>>(tup_lam);
        //args.resize(app->num_args());
        std::vector<const Def*> new_args;
        std::vector<const Def*> new_doms;

        for (size_t i = 0, e = app->num_args(); i != e; ++i) {
            auto a = tup_lam->num_outs();
            if (a == 0) continue; // remove empty tuples
            if (a == 1) continue; // keep
            if (keep_.contains(tup_lam->var(i))) {
                new_doms.emplace_back(tup_lam->var(i)->type());
                new_args .emplace_back(app->arg(i));
            } else {
                for (size_t j = 0; j != a; ++j) {
                    new_args.emplace_back(proj(tup_lam->var(i), a, j));
                    new_doms.emplace_back(proj(tup_lam->dom  (i), a, j));
                }
            }
        }
    }

    return def;
}

undo_t Scalerize::analyze(const Def* def) {
    auto curr_lam = descend<Lam>(def);
    if (curr_lam == nullptr) return No_Undo;

    return No_Undo;
    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nom<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}'", lam);
            auto [undo, _] = put(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(op));

            if (auto lam = op->isa_nom<Lam>(); !ignore(lam) && keep_.emplace(lam).second) {
            }
        }

        return undo;
    }

    return No_Undo;
}

}
#endif
