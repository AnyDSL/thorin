#include "lower_control.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct ClassifyControl : Rewriter {
    ClassifyControl(World& src, World& dst) : Rewriter(src, dst), forest_(src) {}

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto oapp = odef->isa<App>()) {
            auto ocallee = oapp->callee()->isa_nom<Continuation>();
            if (ocallee && ocallee->intrinsic() == Intrinsic::Control) {
                auto obody = oapp->arg(1)->as_nom<Continuation>();
                auto otoken_param = obody->param(1);
                const JoinPointType* jpt = otoken_param->type()->as<JoinPointType>();
                bool leaks = false;
                for (auto use : otoken_param->uses()) {
                    if (use.def()->isa<App>() && use.index() == App::CALLEE_POSITION)
                        continue;
                    src().VLOG("In control block {}, the body continuation {} leaks the control token. Cannot promote to static_control!", oapp, obody);
                    leaks = true;
                    break;
                }

                if (!leaks) {
                    src().VLOG("In control block {}, the body continuation {} does not leak the control token. Promoting to static_control!", oapp, obody);
                    auto nmem = instantiate(oapp->arg(0));
                    auto nbody = instantiate(oapp->arg(1));
                    auto npost = instantiate(oapp->arg(2));
                    return dst().app(instantiate(oapp->filter())->as<Filter>(), dst().static_control(instantiate(jpt)->as<JoinPointType>()->types().skip_front()), { nmem, nbody, npost });
                }
            }
        }

        return Rewriter::rewrite(odef);
    }

    ScopesForest forest_;
};

void classify_control(Thorin& thorin) {
    auto& src = thorin.world_container();
    std::unique_ptr<World> dst = std::make_unique<World>(*src);
    ClassifyControl pass(*src, *dst);
    for (auto e : src->externals())
        pass.instantiate(e.second);
    src.swap(dst);
}

}
