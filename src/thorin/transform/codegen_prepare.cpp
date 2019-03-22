#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

static bool replace_param(const Def* param, const Def* replace_with) {
    bool dirty = false;
    for (auto use : param->copy_uses()) {
        if (auto app = use->isa<App>()) {
            // The return parameter is used either as an
            // argument or as a callee of an App node
            for (auto app_use : app->copy_uses()) {
                if (auto lam = app_use->isa_nominal<Lam>()) {
                    // Do not change replace_with
                    if (lam == replace_with)
                        continue;
                    dirty = true;
                    if (use.index() == 0) {
                        // Callee
                        lam->app(replace_with, app->op(1));
                    } else {
                        // Argument
                        assert(use.index() == 1);
                        lam->app(app->op(0), replace_with);
                    }
                }
            }
        } else if (!use->isa_nominal()) {
            auto& world = use->world();
            Array<const Def*> new_ops(use->ops());
            new_ops[use.index()] = replace_with;
            dirty |= replace_param(use, use->rebuild(world, use->type(), new_ops));
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

        // Rebuild a new parameter to pass to functions using the parameter as-is
        // (i.e without extracting the return continuation). Note that this assumes that
        // the return continuation is the last element of the parameter.
        std::vector<const Def*> ops;
        auto param = scope.entry()->param();
        for (size_t i = 0, n = param->type()->num_ops(); i < n; ++i) {
            auto op = scope.entry()->param(i);
            ops.push_back(op);
        }
        ops.back() = ret_cont;
        auto new_param = world.tuple(ops);

        if (replace_param(ret_param, ret_cont) || replace_param(param, new_param))
            scope.update();
    });
    VLOG("end codegen_prepare");
}

}
