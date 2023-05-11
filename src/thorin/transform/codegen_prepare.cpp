#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

struct CodegenPrepare : public Rewriter {
    CodegenPrepare(World& src, World& dst) : Rewriter(src, dst) {}

    DefMap<Continuation*> wrappers;

    void make_wrapper(const Def* old_return_param) {
        assert(old_return_param);
        assert(!wrappers.contains(old_return_param));
        auto npi = instantiate(old_return_param->type())->as<FnType>();
        npi = dst().fn_type(npi->types());
        auto wrapper = dst().continuation(npi, old_return_param->debug());
        wrappers[old_return_param] = wrapper;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto app = odef->isa<App>()) {
            auto new_ops = Array<const Def*>(app->num_args(), [&](size_t i) -> const Def* {
                if (auto wrapped = wrappers.lookup(app->arg(i)); wrapped.has_value()) {
                    // because wrappers bodies need the rewritten ret param, they need to be created lazily
                    // otherwise, rewriting the param would cause the whole scope to be rewritten first, before we get a chance to put anything in the map
                    if (!(*wrapped)->has_body()) {
                        auto imported_param = instantiate(app->arg(i));
                        (*wrapped)->jump(imported_param, (*wrapped)->params_as_defs(), imported_param->debug());
                    }
                    return dst().return_point(*wrapped);
                } else {
                    return instantiate(app->arg(i));
                }
            });
            return dst().app(instantiate(app->filter())->as<Filter>(), instantiate(app->callee()), new_ops);
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

    // step one: scan for top-level continuations and register their ret params
    ScopesForest(src).for_each([&](Scope& scope) {
        auto ret_param = scope.entry()->ret_param();
        if (ret_param)
            pass.make_wrapper(ret_param);
    });

    // step two: rewrite the world
    for (auto& external : src.externals())
        pass.instantiate(external.second);

    thorin.world_container().swap(destination);
    thorin.world().VLOG("end codegen_prepare");
}

}
