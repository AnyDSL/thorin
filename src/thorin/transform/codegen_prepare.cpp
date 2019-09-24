#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

void codegen_prepare(World& world) {
    VLOG("start codegen_prepare");

    const Param* old_param = nullptr;
    const Def* new_param = nullptr;
    Scope::for_each_rewrite(world,
        [&](const Scope& scope) {
            if (auto entry = scope.entry()->isa<Lam>()) {
                DLOG("scope: {}", entry);
                // new wrapper that calls the return continuation
                auto ret_param = entry->ret_param();
                auto ret_cont = world.lam(ret_param->type()->as<Pi>(), ret_param->debug());
                ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());

                // rebuild a new "param" that substitutes the actual ret_param with ret_cont
                // note that this assumes that the return continuation is the last element of the parameter
                auto ops = entry->param()->split();
                ops.back() = ret_cont;
                new_param = world.tuple(ops);
                return true;
            }
            return false;
        },
        [&](const Def* old_def) -> const Def* {
            if (old_def == old_param) return new_param;
            return nullptr;
        });

    VLOG("end codegen_prepare");
}

}
