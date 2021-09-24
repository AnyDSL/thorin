#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

void codegen_prepare(World& world) {
    world.VLOG("start codegen_prepare");
    Scope::for_each(world, [&](Scope& scope) {
        world.DLOG("scope: {}", scope.entry());
        bool dirty = false;
        auto ret_param = scope.entry()->ret_param();
        auto ret_cont = world.continuation(ret_param->type()->as<FnType>(), ret_param->debug());
        ret_cont->jump(ret_param, ret_cont->params_as_defs(), ret_param->debug());

        for (auto use : ret_param->copy_uses()) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (use.index() != 0) {
                    ucontinuation->update_op(use.index(), ret_cont);
                    dirty = true;
                }
            }
        }

        if (dirty)
            scope.update();
    });
    world.VLOG("end codegen_prepare");
}

}
