#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

struct CodegenPrepare : public Rewriter {
    CodegenPrepare(World& src, World& dst) : Rewriter(src, dst) {}

    DefMap<Continuation*> wrappers;

    Continuation* make_wrapper(const Def* old_return_param) {
        assert(old_return_param);
        if (auto found = wrappers.lookup(old_return_param))
            return *found;
        auto npi = instantiate(old_return_param->type())->as<FnType>();
        npi = dst().fn_type(npi->types());
        auto wrapper = dst().continuation(npi, old_return_param->debug());
        wrappers[old_return_param] = wrapper;
        return wrapper;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto app = odef->isa<App>()) {
            auto new_ops = Array<const Def*>(app->num_args(), [&](size_t i) -> const Def* {
                auto op = app->arg(i);
                if (op->isa<Param>() && op->type()->isa<ReturnType>()) {
                    // because wrappers bodies need the rewritten ret param, they need to be created lazily
                    // otherwise, rewriting the param would cause the whole scope to be rewritten first, before we get a chance to put anything in the map
                    auto wrapped = make_wrapper(op);
                    if (!(wrapped)->has_body()) {
                        auto imported_param = instantiate(app->arg(i));
                        wrapped->jump(imported_param, wrapped->params_as_defs(), imported_param->debug());
                    }
                    return dst().return_point(wrapped);
                } else {
                    return instantiate(app->arg(i));
                }
            });
            return dst().app(instantiate(app->callee()), new_ops);
        }
        return Rewriter::rewrite(odef);
    }
};

/// this pass makes sure the return param is only called directly, by eta-expanding any uses where it appears in another position
// TODO: this effectively prevents tail-calls, this shouldn't run if the backend supports tail-calls
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
