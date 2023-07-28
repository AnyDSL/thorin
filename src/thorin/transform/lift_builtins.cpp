#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_pipeline(World& world) {
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
            auto pipeline_continue = world.continuation(p_cont_type, Intrinsic::PipelineContinue, Debug("pipeline_continue"));
            auto continue_wrapper = world.continuation(cont_type, Debug("continue_wrapper"));
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

bool lift_accelerator_body(Thorin& thorin, Continuation* cont) {
    World& world = thorin.world();
    ScopesForest forest(world);

    static const int inline_threshold = 4;
    if (is_passed_to_intrinsic(cont, Intrinsic::Vectorize)) {
        Scope& scope = forest.get_scope(cont);
        force_inline(scope, inline_threshold);
    }

    Scope& scope = forest.get_scope(cont);

    // remove all continuations - they should be top-level functions and can thus be ignored
    std::vector<const Def*> defs;
    for (auto param : spillable_free_defs(forest, cont)) {
        if (param->isa_nom<Continuation>()) {
            // TODO: assert is actually top level
        } else if (!param->isa<Filter>()) { // don't lift the filter
            assert(!param->isa<App>() && "an app should not be free");
            //assert(param->order() == 0 && "creating a higher-order function");
            defs.push_back(param);
        }
    }

    if (defs.empty())
        return false;

    auto lifted = lift(scope, scope.entry(), defs);
    for (auto use : cont->copy_uses()) {
        if (auto uapp = use->isa<App>()) {
            if (auto callee = uapp->callee()->isa_nom<Continuation>()) {
                if (callee->is_intrinsic()) {
                    auto old_ops = uapp->ops();
                    Array<const Def*> new_ops(old_ops.size() + defs.size());
                    std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin())); // old ops + former free defs
                    assert(old_ops[use.index()] == cont);
                    new_ops[use.index()] = lifted;

                    // jump to new top-level dummy function with new args
                    auto fn_type = world.fn_type(Array<const Type*>(new_ops.size()-App::ARGS_START_POSITION, [&] (auto i) { return new_ops[i+App::ARGS_START_POSITION]->type(); }));
                    auto ncontinuation = world.continuation(fn_type, callee->attributes(), callee->debug());

                    new_ops[App::CALLEE_POSITION] = ncontinuation;
                    uapp->replace_uses(world.app(new_ops[App::CALLEE_POSITION], new_ops.skip_front(2)));
                }
            }
        }
    }

    thorin.cleanup();
    return true;
}

void lift_builtins(Thorin& thorin) {
    // This must be run first
    lift_pipeline(thorin.world());

    bool todo = true;
    while (todo) {
        World& world = thorin.world();
        todo = false;
        for (auto cont : world.copy_continuations()) {
            if (cont->order() <= 1)
                continue;
            if (is_passed_to_accelerator(cont, false)) {
                if (lift_accelerator_body(thorin, cont)) {
                    todo = true;
                    break;
                }
                continue;
            }
        }
    }
}

}
