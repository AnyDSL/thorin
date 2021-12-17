#include "thorin/pass/rw/ret_wrap.h"

namespace thorin {

void RetWrap::enter() {
    auto ret_var = curr_nom()->ret_var();
    if (!ret_var) return;

    // new wrapper that calls the return continuation
    auto ret_cont = world().nom_lam(ret_var->type()->as<Pi>(), ret_var->dbg());
    ret_cont->app(ret_var, ret_cont->var(), ret_var->dbg());

    // rebuild a new "var" that substitutes the actual ret_var with ret_cont
    auto new_vars = curr_nom()->vars();
    assert(new_vars.back() == ret_var && "we assume that the last element is the ret_var");
    new_vars.back() = ret_cont;
    auto new_var = world().tuple(curr_nom()->dom(), new_vars);
    curr_nom()->set(curr_nom()->apply(new_var));
}

}
