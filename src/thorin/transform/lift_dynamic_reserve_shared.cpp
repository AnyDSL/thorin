#include "lift_dynamic_reserve_shared.h"

#include "thorin/analyses/scope.h"

namespace thorin {

void lift_dynamic_reserve_shared(Thorin& thorin) {
    auto& world = thorin.world();

    // Continuation* kernel, const App* launch_call

    ScopesForest forest(world);

    for (auto continuation: world.copy_continuations()) {
        Intrinsic intrinsic = Intrinsic::None;
        const App* launch_call = nullptr;
        visit_capturing_intrinsics(continuation, [&](Continuation* intrinsic_cont, const App* app) {
            if (intrinsic_cont->is_offload_intrinsic()) {
                intrinsic = intrinsic_cont->intrinsic();
                launch_call = app;
                return true;
            }
            return false;
        });

        if (intrinsic == Intrinsic::None)
            continue;

        auto kernel = continuation;
        auto& scope = forest.get_scope(kernel);
        for (auto def: scope.defs()) {
            if (auto app = def->isa<App>()) {
                auto callee = app->callee()->isa<Continuation>();
                if (!callee || callee->intrinsic() != Intrinsic::Reserve)
                    continue;

                auto size = app->arg(1);
                // you don't need to lift literals or stuff computed within the kernel
                if (size->isa<Literal>())
                    continue;

                auto ret = callee->ret_param()->type()->as<FnType>()->types()[1];
                auto param = kernel->append_param(ret);

                auto napp = world.app(app->arg(2), { app->arg(0), param });
                app->replace_uses(napp);

                Array<const Def*> new_launch_args(launch_call->args().size() + 1);
                for (size_t i = 0; i < launch_call->args().size(); i++)
                    new_launch_args[i] = launch_call->arg(i);
                new_launch_args.back() = size;

                auto and_blackjack = launch_call->callee()->as<Continuation>();
                // auto with_hookers = world.continuation(and_blackjack->type(), and_blackjack->attributes(), {});
                auto with_hookers = world.continuation();
                with_hookers->attributes_ = and_blackjack->attributes();
                for (auto arg: new_launch_args)
                    with_hookers->append_param(arg->type());
                auto napp2 = world.app(with_hookers, new_launch_args);
                launch_call->replace_uses(napp2);
            }
        }
    };
}

}
