#include "thorin/pass/rw/ret_wrap.h"

namespace thorin {

void RetWrap::enter(Def* cur_nom) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto ret_param = cur_lam->ret_param()) {
            // new wrapper that calls the return continuation
            auto ret_cont = world().lam(ret_param->type()->as<Pi>(), ret_param->debug());
            ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());

            // rebuild a new "param" that substitutes the actual ret_param with ret_cont
            auto new_params = cur_lam->param()->split();
            assert(new_params.back() == ret_param && "we assume that the last element is the ret_param");
            new_params.back() = ret_cont;
            auto new_param = world().tuple(cur_lam->domain(), new_params);
            old2new_[cur_lam->param()] = new_param;
            ret_conts_.emplace(ret_cont);
        } else if (ret_conts_.contains(cur_lam)) {
            man().map(cur_lam->body(), cur_lam->body());
        }
    }
}

const Def* RetWrap::rewrite(Def* cur_nom, const Def* def) {
    if (cur_nom->isa<Lam>()) {
        if (auto param = def->isa<Param>()) {
            if (auto new_param = old2new_.lookup(param)) return man().map(*new_param, *new_param);
        }
    }

    return def;
}

}
