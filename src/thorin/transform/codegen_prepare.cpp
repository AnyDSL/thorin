#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

struct CodegenPrepare : public Rewriter {
    CodegenPrepare(World& src, World& dst) : Rewriter(src, dst) {}

    Continuation* make_wrapper(const Def* old_return_param) {
        assert(old_return_param);
        auto npi = instantiate(old_return_param->type())->as<FnType>();
        npi = dst().fn_type(npi->types());
        auto wrapper = dst().continuation(npi, old_return_param->debug());
        return wrapper;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto app = odef->isa<App>()) {
            auto new_ops = Array<const Def*>(app->num_args(), [&](size_t i) -> const Def* {
                auto oarg = app->arg(i);
                if (oarg->type()->isa<ReturnType>()) {
                    if (!oarg->isa<ReturnPoint>()) {
                        auto wrapped = make_wrapper(oarg);
                        insert(oarg, dst().return_point(wrapped));
                        auto imported_param = instantiate(app->arg(i));
                        wrapped->jump(imported_param, wrapped->params_as_defs(), imported_param->debug());
                        return dst().return_point(wrapped);
                    }
                }
                return instantiate(app->arg(i));
            });
            return dst().app(instantiate(app->callee()), new_ops);
        }
        return Rewriter::rewrite(odef);
    }
};

/// this pass makes sure the return param is only called directly, by eta-expanding any uses where it appears in another position
void codegen_prepare(Thorin& thorin) {
    thorin.world().VLOG("start codegen_prepare");
    auto& src = thorin.world();
    auto destination = std::make_unique<World>(src);
    CodegenPrepare pass(src, *destination.get());

    for (auto& external : src.externals())
        pass.instantiate(external.second);

    thorin.world_container().swap(destination);
    thorin.world().VLOG("end codegen_prepare");
}

}
