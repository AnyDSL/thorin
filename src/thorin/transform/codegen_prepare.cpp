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
        auto ret_cont = world.continuation(ret_param->type()->as<FnType>(), ret_param->debug());
        ret_cont->jump(ret_param, ret_cont->param(), ret_param->debug());

        for (auto use : ret_param->copy_uses()) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (use.index() != 0) {
                    assert(use.index() == 1);
                    ucontinuation->update_arg(ret_cont);
                    dirty = true;
                }
            } else if (auto tuple = use->isa<Tuple>()) {
                Array<const Def*> ops(tuple->ops());
                ops[use.index()] = ret_cont;
                for (auto use : tuple->uses()) {
                    if (auto ucontinuation = use->isa_continuation()) {
                        ucontinuation->update_arg(world.tuple(ops));
                        dirty = true;
                    }
                }
            }
        }

        for (auto use : scope.entry()->param()->copy_uses()) {
            if (auto ucontinuation = use->isa_continuation()) {
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

                    ucontinuation->update_arg(world.tuple(args));
                    dirty = true;
                }
            }
        }

        if (dirty)
            scope.update();
    });
    VLOG("end codegen_prepare");
}

}
