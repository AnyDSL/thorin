#include "lift2cff.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct Lift2CffRewriter : Rewriter {
    Lift2CffRewriter(World& src, World& dst) : Rewriter(src, dst), forest_(src) {
    }

    Continuation* lambda_lift(Continuation* cont) {
        Scope scope(cont);
        assert(cont->type()->is_returning());

        std::vector<const Def*> lifted_args;
        for (size_t i = 0; i < cont->num_params(); i++)
            lifted_args.push_back(cont->param(i));
        std::vector<const Def*> defs;
        for (auto free : spillable_free_defs(scope)) {
            //if (free == cont)
            //    continue;
            if (auto free_cont = free->isa<Continuation>())
                continue;
            //  free = src().closure(src().closure_type(free_cont->type()->types()), free_cont, src().tuple({}));
            lifted_args.push_back(free);
            defs.push_back(free);
        }
        if (defs.size() > 0) {
            dst().VLOG("Lambda lifting {} ({} free variables)", cont, defs.size());
            auto lifted = lift(scope, defs);
            lifted->set_name(lifted->name() + "_lifted");
            cont->jump(lifted, lifted_args);
            cont->set_filter(cont->all_true_filter());
            cont->set_name(cont->name() + "_unlifted");
            return cont;
        }
        return cont;
    }

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto ocont = odef->isa_nom<Continuation>()) {
            Continuation* ofn = ocont;
            while (ofn) {
                // we found the enclosing function
                if (ofn->type()->is_returning())
                    break;
                Scope& scope = forest_.get_scope(ofn);
                ofn = scope.parent_scope();
            }

            Scope& scope = forest_.get_scope(ofn);
            // we have a function but it's not top-level yet!
            if (scope.parent_scope()) {
                ocont = lambda_lift(ocont);
            }
            auto r = Rewriter::rewrite(ocont);
            //if (ocont != odef)
            //    insert(odef, r);
            return r;
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
