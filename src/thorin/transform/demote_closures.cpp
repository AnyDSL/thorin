#include "demote_closures.h"

namespace thorin {

struct ClosureDemoter {
    ClosureDemoter(World& world) : world_(world) {}

    void run() {
        for (auto cont : world_.copy_continuations()) {
            if (!cont->has_body())
                continue;
            auto app = cont->body();
            if (auto closure = app->callee()->isa_nom<Closure>())
                run(closure);
        }
    }

    void run(Closure* closure) {
        if (processed_.contains(closure)) {
            return;
        }
        processed_.insert(closure);
        if (closure->fn()->has_body()) {
            // these closures will get dealt with inside World::app
            if (closure->self_param() < 0)
                return;

            bool closure_needed = false;
            std::vector<const App*> called_directly;
            for (auto use : closure->uses()) {
                if (auto app = use.def()->isa<App>(); app && use.index() == App::Ops::Callee) {
                    called_directly.push_back(app);
                    continue;
                }
                //if (auto app = use.def()->isa<App>(); app && app->callee() == closure->fn() && (int) use.index() == closure->self_param()) {
                //    continue;
                //}
                closure_needed = true;
            }

            // if the closure is never called directly, we'd be wasting our time
            if (!called_directly.empty()) {
                bool self_param_ok = true;
                const Param* self_param = closure->fn()->param(closure->self_param());
                const ClosureEnv* env = nullptr;
                for (auto use : self_param->uses()) {
                    // the closure argument can be used, but only to extract the environment!
                    if (auto e = use.def()->isa<ClosureEnv>()) {
                        assert(!env);
                        env = e;
                        continue;
                    }
                    self_param_ok = false;
                    break;
                }

                if (self_param_ok && !closure_needed) {
                    //world_.VLOG("demote_closures: {} is called directly, which we can simplify", closure);
                    //todo_ = true;

                    // the wrapper type has the same params as the closure
                    auto wrapper = world_.continuation(world_.fn_type(closure->type()->types()), closure->fn()->debug());

                    // the regular params just get forwarded
                    std::vector<const Def*> wrapper_args;
                    for (auto p: wrapper->params_as_defs())
                        wrapper_args.push_back(p);

                    auto dummy_closure = world_.closure((closure->type())->as<ClosureType>());
                    // the dummy closure environment is a new wrapper param (closure lives in wrapper scope)
                    auto env_param = wrapper->append_param(closure->env()->type());
                    dummy_closure->set_env(env_param);

                    auto fn = closure->fn();

                    // if the closure isn't needed at all, get rid of the self param uses
                    // next round of dead param elimination will do it in
                    if (!closure_needed) {
                        if (env)
                            env->replace_uses(world_.tuple({env->mem(), env_param}));
                        auto dead_fn = world_.continuation(closure->fn()->type());
                        dummy_closure->set_fn(dead_fn, closure->self_param());
                        closure->unset_op(0);
                        closure->set_fn(dead_fn, closure->self_param());
                    } else {
                        // the dummy closure has a dummy function and no self param
                        // TODO: this duplicates the closure and that's no good
                        // (later replace_uses screws up the scope of one of them)
                        dummy_closure->set_fn(closure->fn(), closure->self_param());
                    }

                    // if we had a self param, make sure we insert the closure where that was
                    wrapper_args.insert(wrapper_args.begin() + closure->self_param(), dummy_closure);

                    world_.VLOG("demoted closure {}", closure);
                    wrapper->jump(world_.run(fn), wrapper_args);

                    replace_calls(called_directly, closure, wrapper);
                }
            }
        }
    }

    void replace_calls(std::vector<const App*>& calls, const Closure* closure, Continuation* wrapper) {
        for (auto app : calls) {
            world_.VLOG("demote_closures: {} calls closure {} which only consumes its environment, replacing with wrapper {}", app, closure, wrapper);
            app->replace_uses(world_.app(wrapper, concat(static_cast<ArrayRef<const Def*>>(app->args()), closure->env()), app->debug()));
            todo_ = true;
        }
    }

    World& world_;
    bool todo_ = false;
    DefSet processed_;
};

bool demote_closures(World& world) {
    ClosureDemoter pass(world);
    pass.run();
    return pass.todo_;
}

}