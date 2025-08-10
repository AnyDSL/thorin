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
            //assert(!lookup(closure->fn()) && "fn can't be rewritten yet");
            bool only_called = true;
            for (auto use : closure->uses()) {
                if (use.def()->isa<App>() && use.index() == App::Ops::Callee)
                    continue;
                only_called = false;
                break;
            }
            if (only_called) {
                bool self_param_ok = true;
                auto self_param = closure->fn()->params().back();
                for (auto use : self_param->uses()) {
                    // the closure argument can be used, but only to extract the environment!
                    //if (auto extract = use.def()->isa<Extract>(); extract && is_primlit(extract->index(), 1))
                    //    continue;
                    if (use.def()->isa<ClosureEnv>())
                        continue;
                    self_param_ok = false;
                    break;
                }
                auto fn_t = (closure->fn()->type())->as<FnType>();
                auto types = fn_t->copy_types();//.skip_back(1);
                fn_t = world_.fn_type(types.skip_back(1));

                world_.VLOG("simplify: eliminating closure {} as it is never passed as an argument, and is not recursive", closure);
                todo_ = true;

                auto wrapper = world_.continuation(fn_t, closure->fn()->debug());
                //Array<const Def*> args(closure->fn()->num_params());
                auto dummy_closure = world_.closure((closure->type())->as<ClosureType>());
                dummy_closure->set_fn(world_.continuation((closure->fn()->type())->as<FnType>()));
                auto env_param = wrapper->append_param(closure->env()->type());
                dummy_closure->set_env(env_param);
                wrapper->jump(world_.run((closure->fn())), concat(wrapper->params_as_defs().skip_back(1), static_cast<const Def*>(dummy_closure)));

                replace_calls(closure, wrapper);
            }
        }
    }

    void replace_calls(const Closure* closure, Continuation* wrapper) {
        for (auto use : closure->copy_uses()) {
            if (auto app = use.def()->isa<App>()) {
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