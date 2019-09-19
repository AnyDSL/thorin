#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

void codegen_prepare(World& world) {
    VLOG("start codegen_prepare");
    Scope::for_each(world, [&](Scope& scope) {
        DLOG("scope: {}", scope.entry());
        // new wrapper that calls the return continuation
        auto ret_param = scope.entry()->ret_param();
        auto ret_cont = world.lam(ret_param->type()->as<Pi>(), ret_param->debug());
        ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());

        // rebuild a new "param" that substitutes the actual ret_param with ret_cont
        // note that this assumes that the return continuation is the last element of the parameter
        auto old_param = scope.entry()->param();
        auto ops = old_param->split();
        ops.back() = ret_cont;
        auto new_param = world.tuple(ops);

        auto old_body = scope.entry()->body();
        auto new_body = rewrite(old_body, old_param, new_param, &scope);
        if (new_body != old_body) {
            scope.entry()->set_body(new_body);
            scope.update();
        }
    });
    VLOG("end codegen_prepare");
}

}
