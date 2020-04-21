#if 0
#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"

namespace thorin {

/*
 * helpers
 */

//    ⊥ y
//   +---
// ⊥ |⊥ y
// x |x x - if x == y, else ⊤
// ⊤ |⊤ ⊤
bool CopyProp::join(const Def*& a, const Def* b) {
    if (a->isa<Top>() || b->isa<Bot>() || a == b) return false;

    if (a->isa<Bot>())
        a = b;
    else
        a = world().top(b->type());

    return true;
}

/*
 * PassMan hooks
 */

Def* CopyProp::inspect(Def* def) {
    /*
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info_[old_lam];
        if (preds_n_.contains(old_lam)) info.lattice = Info::PredsN;
        if (keep_   .contains(old_lam)) info.lattice = Info::Keep;

        if (old_lam->is_external() || old_lam->intrinsic() != Lam::Intrinsic::None) {
            info.lattice = Info::Keep;
        } else if (info.lattice != Info::Keep) {
            auto& phis = lam2phis_[old_lam];

            if (info.lattice == Info::PredsN && !phis.empty()) {
                std::vector<const Def*> types;
                for (auto i = phis.begin(); i != phis.end();) {
                    auto proxy = *i;
                    if (keep_.contains(proxy)) {
                        i = phis.erase(i);
                    } else {
                        types.emplace_back(proxy_type(proxy));
                        ++i;
                    }
                }
                //Array<const Def*> types(phis.size(), [&](auto) { return proxy_type(*phi++); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                world().DLOG("new_lam: {} -> {}", old_lam, new_lam);
                new2old_[new_lam] = old_lam;

                size_t n = new_lam->num_params() - phis.size();
                auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
                man().local_map(old_lam->param(), new_param);

                info.new_lam = new_lam;
                lam2info_[new_lam].lattice = Info::PredsN;
                new_lam->set(old_lam->ops());

                size_t i = 0;
                for (auto phi : phis)
                    set_val(new_lam, phi, new_lam->param(n + i++));

                return new_lam;
            }
        }
    }

    */
    return def;
}

const Def* CopyProp::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !man().outside(lam)) {
            //const auto& info = lam2info_[lam];
            /*
            if (auto new_lam = info.new_lam) {
                auto& phis = lam2phis_[lam];
                auto phi = phis.begin();
                Array<const Def*> args(phis.size(), [&](auto) { return get_val(*phi++); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
            */
        }
    }

    return def;
}

bool CopyProp::analyze(const Def* def) {
    if (def->isa<Param>()) return true;

    if (auto app = def->isa<App>()) {
        if (auto new_lam = app->callee()->isa_nominal<Lam>()) {
            if (auto old_lam = new2old_.lookup(new_lam)) {
                auto& info = lam2info_[*old_lam];
                bool todo = false;
                for (size_t i = 0, e = app->num_args(); i != e; ++i)
                    todo |= join(info.args[i], app->arg(i));
                return !todo;
            }
        }
    }

    return true;
}

void CopyProp::retry() {
    lam2info_.clear();
}

void CopyProp::clear() {
    retry();
    new2old_.clear();
    keep_.clear();
}

}
#endif
