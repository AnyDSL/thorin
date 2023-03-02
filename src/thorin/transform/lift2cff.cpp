#include "lift2cff.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct Lift2CffRewriter {
    Lift2CffRewriter(World& src) : world_(src) {
        Scope::for_each(src, [&](auto& scope) {
            top_level_.insert(scope.entry());
        });
    }

    bool needs_lifting(Continuation* cont) {
        bool top_level = top_level_.contains(cont);
        // we are only interested in lambda lifting things that are not already top level
        if (top_level || !cont->has_body())
            return false;
        for (size_t i = 0; i < cont->num_params(); i++) {
            auto p = cont->param(i);
            //if (p->order() >= 2
            //|| (p->order() == 1 && (p != cont->ret_param() || !p->type()->isa<FnType>())))
            if (p->order() >= 1)
                return true;
        }
        return false;
    }

    bool lambda_lift(Continuation* cont) {
        Scope scope(cont);
        std::vector<const Def*> lifted_args;
        for (size_t i = 0; i < cont->num_params(); i++)
            lifted_args.push_back(cont->param(i));
        std::vector<const Def*> defs;
        for (auto free : free_defs(scope)) {
            if (free->isa_nom<Continuation>()) {
                // TODO: assert is actually top level
            } else if (!free->isa<Filter>()) { // don't lift the filter
                assert(!free->isa<App>() && "an app should not be free");
                //assert(!is_mem(free));
                //todo_ = true;
                lifted_args.push_back(free);
                defs.push_back(free);
            }
        }
        if (defs.size() > 0) {
            bool has_callee_use = false;
            for (auto use : cont->uses()) {
                auto uapp = use->isa<App>();
                if (uapp && use.index() == App::CALLEE_POSITION) {
                    has_callee_use = true;
                }
            }
            if (!has_callee_use)
                return false;
            world_.VLOG("Lambda lifting {} ({} free variables)", cont, defs.size());
            auto lifted = lift(scope, defs);
            lifted->set_name(lifted->name() + "_lifted");
            cont->jump(lifted, lifted_args);
            cont->set_filter(cont->all_true_filter());
            cont->set_name(cont->name() + "_unlifted");
            return true;
        }
        return false;
    }

    bool run() {
        Scope::for_each(world_, [&](auto& scope) {
            for (const CFNode* cfnode : scope.f_cfg().reverse_post_order()) {
                auto cont = cfnode->continuation();
                if (needs_lifting(cont)) {
                    todo_ |= lambda_lift(cont);
                }
            }
        });
        return todo_;
    }

    World& world_;
    bool todo_ = false;
    ContinuationSet top_level_;
};

bool lift2cff(Thorin& thorin) {
    bool todo_ = false, did_something;
    do {
        auto& src = thorin.world_container();
        std::unique_ptr<World> dst = std::make_unique<World>(*src);
        did_something = Lift2CffRewriter(*src).run();
        Importer importer(*src, *dst);
        for (auto e : src->externals())
            importer.import(e.second);
        src.swap(dst);
        todo_ |= did_something;
    } while (did_something);
    return todo_;
}

}
