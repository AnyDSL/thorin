#include "demote_closures.h"

#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

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

                    auto old_fn_uses = fn->copy_uses();
                    if (env) {
                        Scope scope(fn);

                        auto args = Array<const Def*>(fn->num_params());
                        const Def* dummy_closure = world_.bottom(closure->type());
                        if (closure->self_param() >= 0)
                            args[closure->self_param()] = dummy_closure;

                        struct R : public Mangler {
                            explicit R(Scope& s, Defs args, const ClosureEnv* env) : Mangler(s, s.entry(), args), env_(env) {}

                            const Def* rewrite(const thorin::Def* odef) override {
                                if (odef == env_) {
                                    auto env_param = add_param(env_->env_type());
                                    return dst().tuple({ instantiate(env_->mem()), env_param});
                                }
                                return Mangler::rewrite(odef);
                            }

                            const ClosureEnv* env_;
                        } r(scope, args, env);

                        fn = r.mangle();
                    }

                    for (auto use : closure->copy_uses()) {
                        auto app = use->isa<App>();
                        if (!app) continue;
                        world_.VLOG("demote_closures: {} calls closure {} which only consumes its environment, replacing with fn {}", app, closure, fn);
                        auto nargs = concat(app->args(), { closure->env() });
                        app->replace_uses(world_.app(fn, nargs, app->debug()));
                        todo_ = true;
                    }
                }
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