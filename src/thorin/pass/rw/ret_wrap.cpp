#include "thorin/pass/rw/ret_wrap.h"

namespace thorin {

void RetWrap::enter() {
    if (auto cur_lam = cur_nom<Lam>()) {
        if (auto ret_param = cur_lam->ret_param()) {
            // new wrapper that calls the return continuation
            auto ret_cont = world().nom_lam(ret_param->type()->as<Pi>(), ret_param->dbg());
            ret_cont->app(ret_param, ret_cont->param(), ret_param->dbg());

            // rebuild a new "param" that substitutes the actual ret_param with ret_cont
            auto new_params = cur_lam->param()->split(cur_lam->num_params());
            assert(new_params.back() == ret_param && "we assume that the last element is the ret_param");
            new_params.back() = ret_cont;
            auto new_param = world().tuple(cur_lam->dom(), new_params);
            cur_lam->set(cur_lam->apply(new_param));
        }
    }
}

}
