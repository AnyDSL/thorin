#include "lift2cff.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct Lift2CffRewriter : Rewriter {
    Lift2CffRewriter(World& src, World& dst) : Rewriter(src, dst), forest_(src) {}

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto ocont = odef->isa_nom<Continuation>()) {
            if (ocont->type()->is_returning()) {
                // we have a function but it's not top-level yet!
                Scope scope(ocont);
                std::vector<const Def*> nfilter;
                std::vector<const Def*> lifted_args;
                for (size_t i = 0; i < ocont->num_params(); i++) {
                    nfilter.push_back(src().literal_bool(false, {}));
                    lifted_args.push_back(ocont->param(i));
                }
                std::vector<const Def*> defs;
                for (auto free: spillable_free_defs(scope)) {
                    if (free->type()->isa<FnType>()) {
                        // forcefully inline any higher order parameters that we introduce: they necessarily correspond to top-level functions anyway
                        nfilter.push_back(src().literal_bool(true, {}));
                    } else
                        nfilter.push_back(src().literal_bool(false, {}));
                    lifted_args.push_back(free);
                    defs.push_back(free);
                }
                if (defs.size() > 0) {
                    dst().VLOG("Lambda lifting {} ({} free variables)", ocont, defs.size());
                    auto lifted = lift(scope, scope.entry(), defs);
                    lifted->set_name(lifted->name() + "_lifted");
                    lifted->set_filter(src().filter(nfilter));

                    auto ncont = ocont->stub(*this);
                    insert(ocont, ncont);

                    Array<const Def*> nargs(lifted_args.size());
                    for (size_t i = 0; i < nargs.size(); i++)
                        nargs[i] = instantiate(lifted_args[i]);
                    ncont->jump(instantiate(lifted), nargs);
                    ncont->set_name(ocont->name());
                    if (ocont->is_external())
                        dst().make_external(ncont);
                    return ncont;
                }
            }
        }
        return Rewriter::rewrite(odef);
    }

    ScopesForest forest_;
};

bool lift2cff(Thorin& thorin) {
    auto& src = thorin.world_container();
    std::unique_ptr<World> dst = std::make_unique<World>(*src);
    Lift2CffRewriter lifter(*src, *dst);
    for (auto e : src->externals())
        lifter.instantiate(e.second);
    src.swap(dst);
}

}
