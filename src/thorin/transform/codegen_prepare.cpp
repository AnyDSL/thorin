#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

static bool replace_param(const Def* param, const Def* replace_with) {
    bool dirty = false;
    for (auto use : param->copy_uses()) {
        // Check if the use is a call
        if (auto app = use->isa<App>()) {
            // For all uses of the call
            for (auto app_use : app->copy_uses()) {
                if (auto lam = app_use->isa_lam()) {
                    if (lam == replace_with)
                        continue;
                    dirty = true;
                    if (use.index() == 0) {
                        // Replace by a call to the new callee, if it is not the new callee itself
                        lam->app(replace_with, app->op(1));
                    } else {
                        assert(use.index() == 1);
                        lam->app(app->op(0), replace_with);
                    }
                }
            }
        } else if (auto tuple = use->isa<Tuple>()) {
            auto& world = tuple->world();
            Array<const Def*> args(tuple->num_ops());
            for (size_t i = 0, n = tuple->num_ops(); i < n; ++i)
                args[i] = tuple->op(i);
            args[use.index()] = replace_with;
            dirty |= replace_param(tuple, world.tuple(args));
        }
    }
    return dirty;
}

void codegen_prepare(World& world) {
    VLOG("start codegen_prepare");
    Scope::for_each(world, [&](Scope& scope) {
        DLOG("scope: {}", scope.entry());
        auto ret_param = scope.entry()->ret_param();
        auto ret_cont = world.lam(ret_param->type()->as<Pi>(), ret_param->debug());
        ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());
        if (replace_param(ret_param, ret_cont))
            scope.update();
    });
    VLOG("end codegen_prepare");
}

}
