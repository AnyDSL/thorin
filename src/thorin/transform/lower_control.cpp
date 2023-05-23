#include "lower_control.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct LowerControl : Rewriter {
    LowerControl(World& src, World& dst) : Rewriter(src, dst), forest_(src) {}

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto control_type = odef->isa<JoinPointType>()) {
            auto ntype = Rewriter::rewrite(control_type)->as<JoinPointType>();
            return dst().closure_type(ntype->types());
        }

        if (auto oapp = odef->isa<App>()) {
            auto ocallee = oapp->callee()->isa_nom<Continuation>();
            if (ocallee && ocallee->intrinsic() == Intrinsic::Control) {
                auto nmem = instantiate(oapp->arg(0));
                auto nbody = instantiate(oapp->arg(1));
                auto npost = instantiate(oapp->arg(2))->as_nom<Continuation>();
                return dst().app(nbody, { nmem, dst().closure(dst().closure_type(npost->type()->types()), npost, dst().tuple({})) });
            }
        }

        return Rewriter::rewrite(odef);
    }

    ScopesForest forest_;
};

void lower_control(Thorin& thorin) {
    auto& src = thorin.world_container();
    std::unique_ptr<World> dst = std::make_unique<World>(*src);
    LowerControl lifter(*src, *dst);
    for (auto e : src->externals())
        lifter.instantiate(e.second);
    src.swap(dst);
}

}
