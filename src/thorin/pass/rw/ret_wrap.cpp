#include "thorin/pass/rw/ret_wrap.h"

namespace thorin {

void RetWrap::enter(Def* cur_nom) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto ret_param = cur_lam->ret_param()) {
            // new wrapper that calls the return continuation
            auto ret_cont = world().nom_lam(ret_param->type()->as<Pi>(), ret_param->dbg());
            ret_cont->app(ret_param, ret_cont->param(), ret_param->dbg());
            ret_conts_.emplace(ret_cont);

            // rebuild a new "param" that substitutes the actual ret_param with ret_cont
            auto new_params = cur_lam->param()->split(cur_lam->num_params());
            assert(new_params.back() == ret_param && "we assume that the last element is the ret_param");
            new_params.back() = ret_cont;
            auto new_param = world().tuple(cur_lam->domain(), new_params);
            old2new_[cur_lam->param()] = new_param;
        }
    }
}

const Def* RetWrap::rewrite(Def* cur_nom, const Def* old_def, const Def*, Defs, const Def*) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto param = old_def->isa<Param>()) {
            if (auto new_param = old2new_.lookup(param)) return *new_param;
        } else if (ret_conts_.contains(cur_lam)) {
            return old_def;
        }
    }

    return nullptr;
}

}
