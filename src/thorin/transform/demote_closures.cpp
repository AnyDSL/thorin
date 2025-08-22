#include "demote_closures.h"

#include "thorin/analyses/scope.h"

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
                    if (auto app = use.def()->isa<App>(); app && app->callee() == closure->fn() && (int) use.index() == App::Ops::FirstArg +closure->self_param()) {
                        continue;
                    }
                    self_param_ok = false;
                    break;
                }

                if (self_param_ok && !closure_needed) {
                    auto fn = closure->fn();
                    auto env_param = fn->append_param(closure->env()->type());

                    if (env)
                        env->replace_uses(world_.tuple({env->mem(), env_param}));

                    auto old_fn_uses = fn->copy_uses();

                    const Def* dummy_closure = world_.bottom(closure->type());

                    replace_calls(closure->copy_uses(), closure, fn, dummy_closure, closure->env(), env_param, 0);
                    replace_calls(old_fn_uses, closure, fn, dummy_closure, closure->env(), env_param, 1);
                }
            }
        }
    }

    void replace_calls(ArrayRef<Use> old_uses, const Closure* closure, Continuation* wrapper, const Def* dummy, const Def* env, const Def* env_param, int cut_args) {
        Scope s(wrapper);
        Array<const Def*> additional_args = {dummy, env};
        Array<const Def*> additional_args_inside = {dummy, env_param};
        for (auto use : old_uses) {
            auto app = use->isa<App>();
            if (!app) continue;
            world_.VLOG("demote_closures: {} calls closure {} which only consumes its environment, replacing with wrapper {}", app, closure, wrapper);
            auto nargs = concat(app->args().skip_back(cut_args), s.contains(app) ? additional_args_inside : additional_args);
            app->replace_uses(world_.app(wrapper, nargs, app->debug()));
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