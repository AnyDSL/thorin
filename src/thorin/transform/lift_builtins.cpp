#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_pipeline(World& world) {
    for (auto cont : world.copy_continuations()) {
        auto callee = cont->callee()->isa_continuation();
        // Binding to the number of arguments to avoid repeated optimization
        if (callee && callee->intrinsic() == Intrinsic::Pipeline && cont->num_args() == 6) {
            auto cont_type = world.fn_type({ world.mem_type() });
            auto p_cont_type = world.fn_type({ world.mem_type(), cont_type });
            auto body_type = world.fn_type({ world.mem_type(), world.type_qs32() });
            auto pipe_type = world.fn_type({
                world.mem_type(),
                world.type_qs32(),
                world.type_qs32(),
                world.type_qs32(),
                body_type,
                cont_type,
                p_cont_type
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
            auto pipeline_continue = world.continuation(p_cont_type, CC::C, Intrinsic::PipelineContinue, Debug("pipeline_continue"));
            auto continue_wrapper = world.continuation(cont_type, Debug("continue_wrapper"));
            auto new_pipeline = world.continuation(pipe_type, CC::C, Intrinsic::Pipeline, callee->debug());
            auto old_body = cont->arg(4);
            auto body_cont = world.continuation(body_type, old_body->debug());
            cont->jump(new_pipeline, thorin::Defs { cont->arg(0), cont->arg(1), cont->arg(2), cont->arg(3), body_cont, cont->arg(5), pipeline_continue });
            Call call(4);
            call.callee() = old_body;
            call.arg(0) = body_cont->param(0);
            call.arg(1) = body_cont->param(1);
            call.arg(2) = continue_wrapper;
            auto target = drop(call);
            continue_wrapper->jump(pipeline_continue, thorin::Defs { continue_wrapper->param(0), cont->arg(5) });
            body_cont->jump(target->callee(), target->args());
        }
    }

}

void lift_builtins(World& world) {
    // This must be run first
    lift_pipeline(world);

    while (true) {
        Continuation* cur = nullptr;
        Scope::for_each(world, [&] (const Scope& scope) {
            if (cur) return;
            for (auto n : scope.f_cfg().post_order()) {
                if (n->continuation()->order() <= 1)
                    continue;
                if (is_passed_to_accelerator(n->continuation(), false)) {
                    cur = n->continuation();
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

        // remove all continuations - they should be top-level functions and can thus be ignored
        std::vector<const Def*> defs;
        for (auto param : free_defs(scope)) {
            if (!param->isa_continuation()) {
                assert(param->order() == 0 && "creating a higher-order function");
                defs.push_back(param);
            }
        }

        auto lifted = lift(scope, defs);
        for (auto use : cur->copy_uses()) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (auto callee = ucontinuation->callee()->isa_continuation()) {
                    if (callee->is_intrinsic()) {
                        auto old_ops = ucontinuation->ops();
                        Array<const Def*> new_ops(old_ops.size() + defs.size());
                        std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin()));    // old ops + former free defs
                        assert(old_ops[use.index()] == cur);
                        new_ops[use.index()] = world.global(lifted, false, lifted->debug());                                // update to new lifted continuation

                        // jump to new top-level dummy function with new args
                        auto fn_type = world.fn_type(Array<const Type*>(new_ops.size()-1, [&] (auto i) { return new_ops[i+1]->type(); }));
                        auto ncontinuation = world.continuation(fn_type, callee->cc(), callee->intrinsic(), callee->debug());
                        ucontinuation->jump(ncontinuation, new_ops.skip_front(), ucontinuation->jump_debug());
                    }
                }
            }
        }

        world.cleanup();
    }
}

}
