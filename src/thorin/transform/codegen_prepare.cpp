#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

void codegen_prepare(World& world) {
    world.VLOG("start codegen_prepare");
    ScopesForest forest(world);
    forest.for_each([&](Scope& scope) {
        world.DLOG("scope: {}", scope.entry());
        bool dirty = false;
        auto ret_param = scope.entry()->ret_param();
        assert(ret_param && "scopes should have a return parameter");
        auto ret_cont = world.continuation(ret_param->type()->as<FnType>(), ret_param->debug());
        ret_cont->jump(ret_param, ret_cont->params_as_defs(), ret_param->debug());

        for (auto use : ret_param->copy_uses()) {
            if (auto uapp = use->isa<App>()) {
                if (use.index() != 0) {
                    auto nops = uapp->copy_ops();
                    nops[use.index()] = ret_cont;
                    auto napp = uapp->rebuild(world, uapp->type(), nops);
                    uapp->replace_uses(napp);
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
