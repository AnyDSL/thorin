#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_pipeline(World& world) {
    for (auto lam : world.copy_lams()) {
        auto callee = lam->app()->callee()->isa_lam();
        // Binding to the number of arguments to avoid repeated optimization
        if (callee && callee->intrinsic() == Intrinsic::Pipeline && lam->app()->num_args() == 6) {
            auto lam_type = world.cn({ world.mem_type() });
            auto p_lam_type = world.cn({ world.mem_type(), lam_type });
            auto body_type = world.cn({ world.mem_type(), world.type_qs32() });
            auto pipe_type = world.cn({
                world.mem_type(),
                world.type_qs32(),
                world.type_qs32(),
                world.type_qs32(),
                body_type,
                lam_type,
                p_lam_type
            });
            // Transform:
            //
            // f(...)
            //     pipeline(..., pipeline_body, return)
            //
            // pipeline_body(mem: mem, i: i32, ret: fn(mem))
            //     ret(mem)
            //
            // Into:
            //
            // f(...)
            //     new_pipeline(..., pipeline_body, return, pipeline_continue)
            //
            // pipeline_body(mem: mem, i: i32)
            //     continue_wrapper(mem)
            //
            // continue_wrapper(mem: mem)
            //     pipeline_continue(mem, return)
            //
            // Note the use of 'return' as the second argument to pipeline_continue.
            // This is required to encode the dependence of the loop body over the call to pipeline,
            // so that lift_builtins can extract the correct free variables.
            auto pipeline_continue = world.lam(p_lam_type, Intrinsic::PipelineContinue, Debug("pipeline_continue"));
            auto continue_wrapper = world.lam(lam_type, Debug("continue_wrapper"));
            auto new_pipeline = world.lam(pipe_type, Intrinsic::Pipeline, callee->debug());
            auto old_body = lam->app()->arg(4);
            auto body_lam = world.lam(body_type, old_body->debug());
            lam->app(new_pipeline, {
                lam->app()->arg(0),
                lam->app()->arg(1),
                lam->app()->arg(2),
                lam->app()->arg(3),
                body_lam,
                lam->app()->arg(5),
                pipeline_continue
            });
            auto target = drop(world.app(old_body, { body_lam->param(0), body_lam->param(1), continue_wrapper }));
            continue_wrapper->app(pipeline_continue, { continue_wrapper->param(0), lam->app()->arg(5) });
            body_lam->set_body(target);
        }
    }

}

void lift_builtins(World& world) {
    // This must be run first
    lift_pipeline(world);

    while (true) {
        Lam* cur = nullptr;
        Scope::for_each(world, [&] (const Scope& scope) {
            if (cur) return;
            for (auto n : scope.f_cfg().post_order()) {
                if (n->lam()->order() <= 1)
                    continue;
                if (is_passed_to_accelerator(n->lam(), false)) {
                    cur = n->lam();
                    break;
                }
            }
        });

        if (!cur) break;

        static const int inline_threshold = 4;
        if (is_passed_to_intrinsic(cur, Intrinsic::Vectorize)) {
            Scope scope(cur);
            force_inline(scope, inline_threshold);
        }

        Scope scope(cur);

        // remove all lams - they should be top-level functions and can thus be ignored
        std::vector<const Def*> defs;
        for (auto param : free_defs(scope)) {
            if (!param->isa_lam()) {
                assert(param->order() == 0 && "creating a higher-order function");
                defs.push_back(param);
            }
        }

        auto lifted = lift(scope, defs);
        for (auto use : cur->copy_uses()) {
            if (auto ulam = use->isa_lam()) {
                if (auto callee = ulam->app()->callee()->isa_lam()) {
                    if (callee->is_intrinsic()) {
                        auto old_ops = ulam->ops();
                        Array<const Def*> new_ops(old_ops.size() + defs.size());
                        std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin()));    // old ops + former free defs
                        assert(old_ops[use.index()] == cur);
                        new_ops[use.index()] = world.global(lifted, false, lifted->debug());                                // update to new lifted lam

                        // jump to new top-level dummy function with new args
                        auto cn = world.cn(Array<const Type*>(new_ops.size()-1, [&] (auto i) { return new_ops[i+1]->type(); }));
                        auto nlam = world.lam(cn, callee->attributes(), callee->debug());
                        ulam->app(nlam, new_ops.skip_front(), ulam->app()->debug());
                    }
                }
            }
        }

        world.cleanup();
    }
}

}
