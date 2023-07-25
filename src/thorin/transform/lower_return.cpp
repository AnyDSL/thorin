#include "lower_return.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct LowerReturn : Rewriter {
    LowerReturn(World& src, World& dst) : Rewriter(src, dst), forest_(src) {}

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto ocont = odef->isa_nom<Continuation>()) {
            if (ocont->is_external()) {
                auto otype = ocont->type();
                auto oret = otype->ret_param_index();
                Array<const Type*> ntypes(otype->num_ops(), [&](int i ) -> const Type* {
                    auto nty = instantiate(otype->op(i))->as<Type>();
                    if (i == oret) {
                        return dst().return_type(nty->as<FnType>()->types());
                    }
                    return nty;
                });
                auto ncont = ocont->stub(*this, dst().fn_type(ntypes));
                insert(odef, ncont);
                ncont->rebuild_from(*this, ocont);
                return ncont;
            }
        }

        if (auto ret_ty = odef->isa<ReturnType>()) {
            Array<const Type*> ntypes(ret_ty->num_ops(), [&](int i ) -> const Type* { return instantiate(ret_ty->op(i))->as<Type>(); });
            return dst().fn_type(ntypes);
        }

        if (auto ret_pt = odef->isa<ReturnPoint>()) {
            return instantiate(ret_pt->continuation());
        }

        if (auto oapp = odef->isa<App>()) {
            Array<const Def*> nargs(oapp->num_args(), [&](int i ) -> const Def* { return instantiate(oapp->arg(i)); });
            if (auto ocallee = oapp->callee()->isa_nom<Continuation>()) {
                if (ocallee->is_external()) {
                    auto ret_i = ocallee->type()->ret_param_index();
                    if (ret_i >= 0)
                        nargs[ret_i] = dst().return_point(nargs[ret_i]->as_nom<Continuation>());
                }
            }
            return dst().app(instantiate(oapp->filter())->as<Filter>(), instantiate(oapp->callee()), nargs);
        }

        return Rewriter::rewrite(odef);
    }

    ScopesForest forest_;
};

void lower_return(Thorin& thorin) {
    auto& src = thorin.world_container();
    std::unique_ptr<World> dst = std::make_unique<World>(*src);
    LowerReturn lifter(*src, *dst);
    for (auto e : src->externals())
        lifter.instantiate(e.second);
    src.swap(dst);
    thorin.cleanup();
}

}
