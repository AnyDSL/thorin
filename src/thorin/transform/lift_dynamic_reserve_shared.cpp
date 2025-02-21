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

        std::vector<const Def*> new_launch_args(launch_call->args().size());
        for (size_t i = 0; i < launch_call->args().size(); i++)
            new_launch_args[i] = launch_call->arg(i);

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


                new_launch_args.emplace_back(size);
            }
        }

        if (new_launch_args.size() == launch_call->num_args())
            continue;

        auto old_intrinsic = launch_call->callee()->as<Continuation>();
        auto new_intrinsic = world.continuation(old_intrinsic->debug());
        new_intrinsic->attributes_ = old_intrinsic->attributes();
        for (auto arg: new_launch_args)
            new_intrinsic->append_param(arg->type());
        auto napp2 = world.app(new_intrinsic, new_launch_args);
        launch_call->replace_uses(napp2);
    };
}

}
