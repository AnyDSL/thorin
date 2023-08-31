#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void insert_pipeline_continue(Thorin& thorin) {
    World& world = thorin.world();
    for (auto cont : world.copy_continuations()) {
        if (!cont->has_body()) continue;
        auto body = cont->body();
        auto callee = body->callee()->isa_nom<Continuation>();
        // Binding to the number of arguments to avoid repeated optimization
        if (callee && callee->intrinsic() == Intrinsic::Pipeline && body->num_args() == 6) {
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
            auto pipeline_continue = world.continuation(p_cont_type, Intrinsic::PipelineContinue, { "pipeline_continue" });
            auto continue_wrapper = world.continuation(cont_type, { "continue_wrapper" });
            auto new_pipeline = world.continuation(pipe_type, Intrinsic::Pipeline, callee->debug());
            auto old_body = body->arg(4);
            auto body_cont = world.continuation(body_type, old_body->debug());
            cont->jump(new_pipeline, thorin::Defs { body->arg(0), body->arg(1), body->arg(2), body->arg(3), body_cont, body->arg(5), pipeline_continue });
            auto target = drop(old_body, {body_cont->param(0), body_cont->param(1), continue_wrapper});
            assert(target->has_body());
            continue_wrapper->jump(pipeline_continue, thorin::Defs { continue_wrapper->param(0), body->arg(5) });
            body_cont->jump(target->body()->callee(), target->body()->args());
        }
    }

}

}
