#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"
#include "thorin/util/log.h"

namespace thorin {

bool CopyProp::LamInfo::join(const App* app) {
    bool todo = false;
    for (size_t i = 0, e = params.size(); i != e; ++i) {
        auto& lattice = std::get<Lattice   >(params[i]);
        auto& param   = std::get<const Def*>(params[i]);

        if (lattice == Top || app->arg(i)->isa<Bot>()) continue;

        if (param->isa<Bot>()) {
            todo |= param != app->arg(i);
            param = app->arg(i);
        } else if (param == app->arg(i)) {
            /* do nothing */
        } else {
            lattice = Top;
            todo = true;
        }
    }
    return todo;
}

bool CopyProp::set_top(Lam* lam) {
    bool todo = false;
    auto& info = lam2info(lam);
    info.new_lam = nullptr;
    for (auto& param : info.params) {
        todo |= std::get<Lattice>(param) != Top;
        std::get<Lattice>(param) = Top;
    }
    return todo;
}

const Def* CopyProp::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info(lam);
            if (auto new_lam = info.new_lam) {
                std::vector<const Def*> new_args;
                for (size_t i = 0, e = info.params.size(); i != e; ++i) {
                    auto& lattice = std::get<Lattice   >(info.params[i]);
                    auto& param   = std::get<const Def*>(info.params[i]);
                    if (lattice == Top)
                        new_args.emplace_back(app->arg(i));
                    else if (app->arg(i)->isa<Bot>() || app->arg(i) == param)
                        continue;
                    else
                        return app;
                }

                return world().app(new_lam, new_args, app->debug());
            }
        }
    }

    return def;
}

void CopyProp::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        if (old_lam->is_external() || old_lam->intrinsic() != Lam::Intrinsic::None) {
            set_top(old_lam);
        } else {
            auto& info = lam2info(old_lam);
            std::vector<const Def*> new_domain;
            for (size_t i = 0, e = old_lam->num_params(); i != e; ++i) {
                if (std::get<Lattice>(info.params[i]) == Top)
                    new_domain.emplace_back(old_lam->param(i)->type());
            }

            if (new_domain.size() != old_lam->num_params()) {
                man().new_state();

                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                outf("new_lam: {}:{} -> {}:{}", old_lam, old_lam->type(), new_lam, new_lam->type());
                new2old(new_lam) = old_lam;
                info.new_lam = new_lam;
            }
        }
    }
}

void CopyProp::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        if (auto old_lam = new2old(new_lam)) {
            auto& info = lam2info(old_lam);
            size_t n = info.params.size();
            size_t j = 0;
            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) {
                if (std::get<Lattice>(info.params[i]) == Top)
                    return new_lam->param(j++);
                return std::get<const Def*>(info.params[i]);
            }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());
        }
    }
}

void CopyProp::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            if (new2old(lam) == nullptr) {
                auto& info = lam2info(lam);
                if (info.join(app)) {
                    man().undo(info.undo);
                    info.new_lam = nullptr;
                }
            }
        }

        if (auto lam = app->arg()->isa_nominal<Lam>()) {
            if (set_top(lam))
                man().undo(lam2info(lam).undo);
        }
    } else {
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            if (auto lam = def->op(i)->isa_nominal<Lam>())
                if (set_top(lam))
                    man().undo(lam2info(lam).undo);
        }
    }
}

}
