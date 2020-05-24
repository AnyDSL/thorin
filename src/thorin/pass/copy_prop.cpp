#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"

namespace thorin {

void CopyProp::visit(Def* cur_nom, Def* vis_nom) {
    auto   cur_lam = cur_nom->isa<Lam>();
    auto param_lam = vis_nom->isa<Lam>();
    if (!cur_lam || !param_lam || keep_.contains(param_lam)) return;

    param_lam = lam2param(param_lam);
    if (param_lam->is_intrinsic() || param_lam->is_external()) {
        keep_.emplace(param_lam);
        return;
    }

    // build a prop_lam where args has been propagated from param_lam
    auto&& [visit, _] = get<Visit>(param_lam);
    auto& args = args_[param_lam];
    if (auto& prop_lam = visit.prop_lam; !prop_lam) {
        args.resize(param_lam->num_params());
        std::vector<const Def*> types;
        for (size_t i = 0, e = args.size(); i != e; ++i) {
            if (auto arg = args[i]; arg && arg->isa<Top>())
                types.emplace_back(arg->type());
        }

        auto prop_domain = world().sigma(types);
        prop_lam = world().lam(world().pi(prop_domain, param_lam->codomain()), param_lam->debug());
        man().mark_tainted(prop_lam);
        world().DLOG("param_lam => prop_lam: {}: {} => {}: {}", param_lam, param_lam->type()->domain(), prop_lam, prop_domain);
        prop2param_[prop_lam] = param_lam;
    }
}

void CopyProp::enter(Def* nom) {
    auto prop_lam = nom->isa<Lam>();
    if (!prop_lam) return;

    if (auto param_lam = prop2param(prop_lam)) {
        auto& args = args_[param_lam];
        Array<const Def*> new_params(args.size(), [&](size_t i) {
            if (args[i])
                return args[i]->isa<Top>() ? param_lam->param(i) : args[i];
            else
                return world().bot(param_lam->param(i)->type());
        });
        auto new_param = world().tuple(new_params);
        man().map(param_lam->param(), new_param);
        prop_lam->set(param_lam->ops());
    }
}

const Def* CopyProp::rewrite(Def*, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto param_lam = app->callee()->isa_nominal<Lam>()) {
            auto&& [visit, _] = get<Visit>(param_lam);
            auto& args = args_[param_lam];
            if (auto& prop_lam = visit.prop_lam) {
                std::vector<const Def*> new_args;
                std::vector<const Def*> non_args;
                bool use_proxy = false;
                for (size_t i = 0, e = args.size(); i != e; ++i) {
                    if (args[i] == app->arg(i)) {
                        non_args.emplace_back(app->arg(i));
                    } else {
                        new_args.emplace_back(app->arg(i));
                        non_args.emplace_back(world().top(app->arg(i)->type()));
                        use_proxy = true;
                    }
                }

                if (use_proxy) {
                    non_args.emplace_back(param_lam);
                    return proxy(app->type(), non_args);
                } else {
                    return world().app(prop_lam, new_args);
                }
            }
        }
    }

    return def;
}

void join(const Def*& a, const Def* b) {
    if (!a) {
        a = b;
    }else if (a == b) {
    } else {
        a = a->world().top(a->type());
    }
}

undo_t CopyProp::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->isa<Param>()) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto param_lam = proxy->ops().back()->as_nominal<Lam>();
        auto proxy_args = proxy->ops().skip_back(1);
        auto&& [visit, undo_visit] = get<Visit>(param_lam);
        auto& args = args_[param_lam];
        for (size_t i = 0, e = proxy_args.size(); i != e; ++i) {
            auto x = args[i];
            auto xx = x ? x->unique_name() : std::string("<null>");
            join(args[i], proxy_args[i]);
            world().DLOG("{} = {} join {}", args[i], xx, proxy_args[i]);
            visit.prop_lam = nullptr;
        }

        return undo_visit;
    }

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);
        if (auto lam = op->isa_nominal<Lam>()) {
            lam = lam2param(lam);
            auto&& [visit, undo_visit] = get<Visit>(lam);
            // if lam does not occur as callee - we can't do anything
            if ((!def->isa<App>() || i != 0)) {
                if (keep_.emplace(lam).second) {
                    world().DLOG("keep: {}", lam);
                    undo = std::min(undo, undo_visit);
                    visit.prop_lam = nullptr;
                }
            }
        }
    }

    return undo;
}

}
