#include "lift2cff.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/rewrite.h"
#include "thorin/transform/mangle.h"

namespace thorin {

struct Lift2CffRewriter : Rewriter {
    Lift2CffRewriter(World& src, World& dst, bool all) : Rewriter(src, dst), forest_(src), all_(all) {}

    struct Lifted {
        Continuation* ncont;
        std::vector<const Def*> env;
    };

    Lifted& get_lifted_version(Continuation* ocont) {
        if (map_.contains(ocont))
            return *map_[ocont];

        std::vector<const Def*> nfilter;
        std::vector<const Def*> oenv;
        for (size_t i = 0; i < ocont->num_params(); i++) {
            nfilter.push_back(src().literal_bool(false, {}));
            //lifted_args.push_back(ocont->param(i));
        }
        std::vector<const Def*> defs;
        //assert(ocont->name().find("_xlifted") == std::string::npos);
        for (auto free: spillable_free_defs(forest_, ocont)) {
            if (free->type()->isa<FnType>()) {
                assert(false);
                // forcefully inline any higher order parameters that we introduce: they necessarily correspond to top-level functions anyway
                //nfilter.push_back(src().literal_bool(true, {}));
            } //else
            //    nfilter.push_back(src().literal_bool(false, {}));
            oenv.push_back(free);
            defs.push_back(free);
        }
        dst().VLOG("Lambda lifting {} ({} free variables)", ocont, defs.size());
        Scope& scope = forest_.get_scope(ocont);
        auto olifted = lift(scope, scope.entry(), defs);
        dst().VLOG("Resulting in {}", olifted);
        auto ncallee = instantiate(olifted)->as_nom<Continuation>();

        Lifted& lifted = *map_.emplace(ocont, std::make_unique<Lifted>(Lifted { ncallee, {} })).first->second;

        for (auto e : oenv) {
            lifted.env.push_back(instantiate(e));
        }

        return lifted;
    }

    const Def* rewrite(const thorin::Def *odef) override {
        if (auto oapp = odef->isa<App>()) {
            if (auto ocallee = oapp->callee()->isa_nom<Continuation>()) {
                std::vector<const Def*> nargs;
                for (auto oa : oapp->args())
                    nargs.push_back(instantiate(oa));
                Lifted& lifted = get_lifted_version(ocallee);
                for (auto e : lifted.env)
                    nargs.push_back(e);
                return dst().app(lifted.ncont, nargs);
            }
        } else if (auto ocont = odef->isa_nom<Continuation>()) {
            // assert(spillable_free_defs(forest_, ocont).empty());
        }
        /*if (auto ocont = odef->isa_nom<Continuation>()) {
            if (ocont->type()->is_returning() || all_) {
                // we have a function but it's not top-level yet!
                Scope scope(ocont);
                std::vector<const Def*> nfilter;
                std::vector<const Def*> lifted_args;
                for (size_t i = 0; i < ocont->num_params(); i++) {
                    nfilter.push_back(src().literal_bool(false, {}));
                    lifted_args.push_back(ocont->param(i));
                }
                std::vector<const Def*> defs;
                //assert(ocont->name().find("_xlifted") == std::string::npos);
                for (auto free: spillable_free_defs(forest_, ocont)) {
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
                    dst().VLOG("Resulting in {}", lifted);
                    lifted->set_name(lifted->name() + "_xlifted_" + std::to_string(lifted->gid()));
                    //lifted->set_filter(src().filter(nfilter));

                    auto still_free = spillable_free_defs(forest_, lifted);
                    if (still_free.size() > 0) {
                        dst().VLOG("Lifting of {} failed, we still have free variables in the lifted version: {}", ocont, lifted);
                        for (auto sf : still_free) {
                            dst().ELOG("{}", sf);
                        }
                        assert(false);
                    }

                    auto ncont = ocont->stub(*this, instantiate(ocont->type())->as<Type>());
                    insert(ocont, ncont);

                    Continuation* nlifted = lifted->stub(*this, instantiate(lifted->type())->as<Type>());
                    insert(lifted, nlifted);
                    dst().VLOG("Imported as {} {}", nlifted, ncont);
                    nlifted->rebuild_from(*this, lifted);

                    Array<const Def*> nargs(lifted_args.size());
                    for (size_t i = 0; i < nargs.size(); i++)
                        nargs[i] = instantiate(lifted_args[i]);

                    ncont->jump(nlifted, nargs);
                    ncont->set_name(ocont->name());
                    if (ocont->is_external())
                        dst().make_external(ncont);
                    return ncont;
                }
            }
        }*/
        return Rewriter::rewrite(odef);
    }

    ContinuationMap<std::unique_ptr<Lifted>> map_;
    ScopesForest forest_;
    bool all_;
};

bool lift2cff(Thorin& thorin, bool all) {
    auto& src = thorin.world_container();
    std::unique_ptr<World> dst = std::make_unique<World>(*src);
    Lift2CffRewriter lifter(*src, *dst, all);
    for (auto e : src->externals())
        lifter.instantiate(e.second);
    src.swap(dst);
}

}
