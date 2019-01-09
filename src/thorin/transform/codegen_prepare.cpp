#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

void codegen_prepare(World& world) {
    VLOG("start codegen_prepare");
    Scope::for_each(world, [&](Scope& scope) {
        DLOG("scope: {}", scope.entry());
        bool dirty = false;
        auto ret_param = scope.entry()->ret_param();
        auto ret_cont = world.lam(ret_param->type()->as<Pi>(), ret_param->debug());
        ret_cont->app(ret_param, ret_cont->param(), ret_param->debug());

        for (auto use : ret_param->copy_uses()) {
            if (auto ulam = use->isa_lam()) {
                if (use.index() != 0) {
                    assert(use.index() == 1);
                    ulam->set_body(ret_cont);
                    dirty = true;
                }
            } else if (auto tuple = use->isa<Tuple>()) {
                Array<const Def*> ops(tuple->ops());
                ops[use.index()] = ret_cont;
                for (auto use : tuple->uses()) {
                    if (auto ulam = use->isa_lam()) {
                        if (auto app = ulam->app()) {
                            ulam->app(app->callee(), ops, app->debug());
                            dirty = true;
                        }
                    }
                }
            }
        }

        for (auto use : scope.entry()->param()->copy_uses()) {
            if (auto ulam = use->isa_lam()) {
                if (use.index() != 0) {
                    assert(use.index() == 1);
                    Array<const Def*> args(scope.entry()->num_params());
                    size_t i = 0;
                    for (auto param : scope.entry()->params()) {
                        if (param == ret_param)
                            args[i++] = ret_cont;
                        else
                            args[i++] = param;
                    }

                    if (auto app = ulam->app()) {
                        ulam->app(app->callee(), args, app->debug());
                        dirty = true;
                    }
                }
            }
        }

        if (dirty)
            scope.update();
    });
    VLOG("end codegen_prepare");
}

}
