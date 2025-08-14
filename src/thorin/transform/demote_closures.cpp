#include "demote_closures.h"

namespace thorin {

struct ClosureDemoter {
    ClosureDemoter(World& world) : world_(world) {}

    void run() {
        for (auto cont : world_.copy_continuations()) {
            if (!cont->has_body())
                continue;
            auto app = cont->body();
            if (auto closure = app->callee()->isa<Closure>())
                run(closure);
        }
    }

    void run(const Closure* closure) {
        if (processed_.contains(closure)) {
            return;
        }
        processed_.insert(closure);
        if (closure->fn()->has_body()) {
            // these closures will get dealt with inside World::app
            if (closure->self_param() < 0)
                return;

            bool called_directly = false;
            for (auto use : closure->uses()) {
                if (use.def()->isa<App>() && use.index() == App::Ops::Callee) {
                    called_directly = true;
                    break;
                }
            }

            // if the closure is never called directly, we'd be wasting our time
            if (called_directly) {
                bool self_param_ok = true;
                const Param* self_param = closure->fn()->param(closure->self_param());
                for (auto use : self_param->uses()) {
                    // the closure argument can be used, but only to extract the environment!
                    if (use.def()->isa<ClosureEnv>())
                        continue;
                    self_param_ok = false;
                    break;
                }

                if (self_param_ok) {
                    //world_.VLOG("demote_closures: {} is called directly, which we can simplify", closure);
                    //todo_ = true;

                    // the wrapper type has the same params as the closure
                    auto wrapper = world_.continuation(world_.fn_type(closure->type()->types()), closure->fn()->debug());

                    // the regular params just get forwarded
                    std::vector<const Def*> wrapper_args;
                    for (auto p: wrapper->params_as_defs())
                        wrapper_args.push_back(p);

                    auto dummy_closure = world_.closure((closure->type())->as<ClosureType>());
                    // the dummy closure has a dummy function and no self param
                    dummy_closure->set_fn(world_.continuation((closure->fn()->type())->as<FnType>()), -1);
                    // the dummy closure environment is a new wrapper param (closure lives in wrapper scope)
                    auto env_param = wrapper->append_param(closure->env()->type());
                    dummy_closure->set_env(env_param);

                    // if we had a self param, make sure we insert the closure where that was
                    wrapper_args.insert(wrapper_args.begin() + closure->self_param(), dummy_closure);

                    world_.VLOG("simplify: old env: {} new env: {} param: {}", closure->env()->type(), dummy_closure->env()->type(), env_param->type());
                    wrapper->jump(world_.run((closure->fn())), wrapper_args);

                    replace_calls(closure, wrapper);
                }
            }
        }
    }

    void replace_calls(const Closure* closure, Continuation* wrapper) {
        for (auto use : closure->copy_uses()) {
            if (auto app = use.def()->isa<App>(); app && use.index() == App::Ops::Callee) {
                world_.VLOG("demote_closures: {} calls closure {} which only consumes its environment, replacing with wrapper {}", app, closure, wrapper);
                use->replace_uses(world_.app(wrapper, concat(static_cast<ArrayRef<const Def*>>(app->args()), closure->env()), app->debug()));
                todo_ = true;
            }
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