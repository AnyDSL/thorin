#include "thorin/pass/beta_eta_conv.h"

#include "thorin/rewrite.h"

namespace thorin {

static bool is_free(DefSet& done, const Param* param, const Def* def) {
    if (/* over-approximation */ def->isa_nominal() || def == param) return true;
    if (def->isa<Param>()) return false;

    if (done.emplace(def).second) {
        for (auto op : def->ops()) {
            if (is_free(done, param, op)) return true;
        }
    }

    return false;
}

static bool is_free(const Param* param, const Def* def) {
    DefSet done;
    return is_free(done, param, def);
}

BetaEtaConv::Lattice BetaEtaConv::lattice(Lam* lam) {
    if (auto eta_i = lam2eta_.find(lam); eta_i != lam2eta_.end()) {
        switch (eta_i->second.index()) {
            case Glob::Eta_Wrap: return Lattice::Eta_Wrap;
            case Glob::Callee_N: return Lattice::Callee_N;
        }
    }

    switch (auto [loc, _, __] = insert<LamMap<Loc>>(lam); loc) {
        case Loc::Non_Callee_1: return Lattice::Non_Callee_1;
        case Loc::Callee_1:     return Lattice::Callee_1;
        case Loc::Bot:          return Lattice::Bot;
    }

    THORIN_UNREACHABLE;
}

const Def* BetaEtaConv::rewrite(Def*, const Def* def) {
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>(); lam != nullptr && !ignore(lam) && !man().is_tainted(lam)) {
            //  η-reduction: λx.e x -> e, whenever x does not appear free in e
            if (auto app = lam->body()->isa<App>(); app != nullptr && app->arg() == lam->param() && !is_free(lam->param(), app->callee())) {
                auto new_def = def->refine(i, app->callee());
                world().DLOG("eta-reduction '{}' -> '{}'", def, new_def);
                return new_def;
            }

            auto app = def->isa<App>();
            bool callee = app != nullptr && i == 0;
            switch (lattice(lam)) {
                case Lattice::Bot:
                    if (callee) {
                        std::get<Loc&>(insert<LamMap<Loc>>(lam, Loc::Callee_1)) = Loc::Callee_1;
                        world().DLOG("beta-redunction '{}'", lam);
                        return lam->apply(app->arg());
                    }
                    break;
                case Lattice::Callee_1:
                    if (callee) {
                        world().DLOG("proxy app '{}'", lam);
                        return proxy(app->type(), {lam, app->arg()});
                    }
                    break;
                case Lattice::Non_Callee_1: break;
                case Lattice::Callee_N:     break;
                case Lattice::Eta_Wrap:
                    if (!callee) {
                        auto eta = lam2eta_[lam];
                        if (eta == nullptr) {
                            eta = lam->stub(world(), lam->type(), lam->debug());
                            eta->app(lam, eta->param());
                        }
                        //return def->refine(i, eta);
                        return def;
                    }
                    break;
            }
        }
    }

    return def;
}

undo_t BetaEtaConv::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = descend(cur_nom, def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        world().DLOG("found proxy app of '{}'", lam);

        auto&& [loc, undo, _] = insert<LamMap<Loc>>(lam);
        switch (lattice(lam)) {
            case Lattice::Bot:
                THORIN_UNREACHABLE;
            case Lattice::Callee_1:
                lam2eta_[lam].set(nullptr, Glob::Callee_N);
                return undo;
            case Lattice::Non_Callee_1:
                lam2eta_[lam].set(nullptr, Glob::Eta_Wrap);
                return undo;
            case Lattice::Callee_N:
            case Lattice::Eta_Wrap:
                return undo;
        }
    }

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        undo = std::min(undo, analyze(cur_nom, def->op(i)));

        if (auto lam = def->op(i)->isa_nominal<Lam>(); lam != nullptr && !ignore(lam) && !man().is_tainted(lam)) {
            auto&& [loc, u, __] = insert<LamMap<Loc>>(lam);
            auto app = def->isa<App>();
            bool callee = app != nullptr && i == 0;

            switch (lattice(lam)) {
                case Lattice::Bot:
                    assert(!callee);
                    loc = Loc::Non_Callee_1;
                    break;
                case Lattice::Eta_Wrap:
                    break;
                default:
                    if (!callee) {
                        lam2eta_[lam].set(nullptr, Glob::Eta_Wrap);
                        undo = std::min(undo, u);
                    }
                    break;
            }
        }
    }

    return No_Undo;
}

}
