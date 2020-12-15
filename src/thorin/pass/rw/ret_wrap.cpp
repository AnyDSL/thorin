#include "thorin/pass/rw/ret_wrap.h"

namespace thorin {

void RetWrap::enter() {
    if (auto cur_lam = cur_nom<Lam>()) {
        if (auto ret_var = cur_lam->ret_var()) {
            // new wrapper that calls the return continuation
            auto ret_cont = world().nom_lam(ret_var->type()->as<Pi>(), ret_var->dbg());
            ret_cont->app(ret_var, ret_cont->var(), ret_var->dbg());

            // rebuild a new "var" that substitutes the actual ret_var with ret_cont
            auto new_vars = cur_lam->var()->split(cur_lam->num_vars());
            assert(new_vars.back() == ret_var && "we assume that the last element is the ret_var");
            new_vars.back() = ret_cont;
            auto new_var = world().tuple(cur_lam->dom(), new_vars);
            cur_lam->set(cur_lam->apply(new_var));
        }
    }
}

}
