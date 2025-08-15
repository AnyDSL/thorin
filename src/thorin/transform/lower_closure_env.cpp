#include "lower_closure_env.h"

#include "thorin/transform/rewrite.h"

namespace thorin {

struct LowerClosureEnv : public Rewriter {
    LowerClosureEnv(World& src, World& dst) : Rewriter(src, dst) {}

    const Def* rewrite(const thorin::Def* odef) override {
        if (auto oclosure = odef->isa_nom<Closure>()) {
            auto nclosure = dst().closure(instantiate(oclosure->type())->as<ClosureType>(), oclosure->debug());
            insert(oclosure, nclosure);
            nclosure->set_fn(instantiate(oclosure->fn())->as_nom<Continuation>(), oclosure->self_param());
            auto env_type = instantiate(oclosure->env()->type())->as<Type>();
            auto nenv = instantiate(oclosure->env());
            if (is_thin(env_type)) {
                nclosure->set_env(nenv);
            } else {
                dst().WLOG("bad: leaking closure environment for: '{}'", oclosure);
                nclosure->set_env(dst().heap_cell(nenv));
            }
            return nclosure;
        } else if (auto oenv = odef->isa<ClosureEnv>()) {
            auto nclosure = instantiate(oenv->op(1));
            auto nmem = instantiate(oenv->mem());
            auto env_type = instantiate(oenv->env_type())->as<Type>();
            const Def* nenv;
            if (is_thin(env_type)) {
                nenv = dst().bitcast(env_type, dst().extract(nclosure, 1));
            } else {
                auto ptr = dst().bitcast(dst().ptr_type(env_type), dst().extract(nclosure, 1));
                auto load = dst().load(nmem, ptr);
                nenv = load->out(1);
                nmem = load->out(0);
            }

            assert(nenv->type() == env_type);

            return dst().tuple({nmem, nenv});
        }
        return Rewriter::rewrite(odef);
    }
};

void lower_closure_env(thorin::Thorin& thorin) {
    thorin.world().VLOG("start lower_closure_env");
    auto& src = thorin.world();
    auto destination = std::make_unique<World>(src);
    LowerClosureEnv pass(src, *destination.get());

    pass.rewrite_externals();

    thorin.world_container().swap(destination);
    thorin.world().VLOG("end lower_closure_env");
}

}
