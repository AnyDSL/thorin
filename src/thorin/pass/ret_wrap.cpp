#include "thorin/pass/ret_wrap.h"

namespace thorin {

void RetWrap::enter(Def* cur_nom) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto ret_param = cur_lam->ret_param()) {
            // new wrapper that calls the return continuation
            auto ret_cont = world().lam(ret_param->type()->as<Pi>(), ret_param->debug());
            ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());
            ret_param2cont_[ret_param] = ret_cont;
        }
    }
}

const Def* RetWrap::rewrite(Def*, const Def* def) {
    if (auto app = def->isa<App>()) {
        bool update = false;
        Array<const Def*> new_args(app->num_args(), [&](size_t i) {
            if (auto ret_cont = ret_param2cont_.lookup(app->arg(i))) {
                update = true;
                return *ret_cont;
            }
            return app->arg(i);
        });

        if (update) return world().app(app->callee(), new_args, app->debug());
    }

    return def;
}

}
