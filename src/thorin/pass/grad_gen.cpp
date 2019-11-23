#include "thorin/pass/grad_gen.h"


namespace thorin {

    void recur_ops(const Def* def, Stream& stream) {
        stream.indent();
        stream.fmt("-{}: {}", def->node_name(), def).endl();
        for (auto op : def->ops()) {
            if (!op->isa<Lit>()) {
                recur_ops(op, stream);
            }
        }
        stream.dedent();
    }

    const Def* GradGen::rewrite(const Def* def) {
        if (auto grad = isa_grad(def)) {

            errf("DEF\n{}\n", def);

            ////////// Head

            auto grad_lam = world().lam(grad->grad_type, {});
            auto grad_ret = grad_lam->ret_param();
            auto grad_mem = grad_lam->mem_param();
            auto grad_domain = grad_lam->domain();


            ////////// Initialize gradients with one

            auto grad_tangent_types = grad_domain->ops().skip_front().skip_back();
            Array<const Def*> param_tangents(grad_tangent_types.size());
            for (size_t i = 0; i < grad_tangent_types.size(); ++i) {
                param_tangents[i] = world().lit_tangent_one(grad_tangent_types[i]);
            }

            ////////// Create return value

            auto ret_pi_type = grad_ret->type()->as<Pi>();
            auto ret_params_type = ret_pi_type->domain()->ops();
            Array<const Def*> ret_params(ret_params_type.size());
            ret_params[0] = grad_mem;
            for (size_t i = 1; i < ret_params_type.size(); ++i) {
                ret_params[i] = param_tangents[i - 1];
            }

            //THORIN_BREAK;

            auto ret_val = world().tuple(ret_params);

            ////////// Call return continuation

            auto grad_body = world().app(grad_ret, ret_val);
            grad_lam->set_body(grad_body);
            grad_lam->set_filter(world().lit_false());

            //return grad_lam;

        }

        return def;
    }

    std::optional<GradGen::GradInfo> GradGen::isa_grad(const Def* def) const {
        if (auto ds2cps = def->isa<DS2CPS>()) {
            if (auto outer_app = ds2cps->ds()->isa<App>(); outer_app->num_args() > 0) {
                if (auto cps2ds = outer_app->arg(0)->isa<CPS2DS>()) {
                    auto lam = cps2ds->cps()->as<Lam>();
                    if (auto app = outer_app->callee()->isa<App>()) {
                        if (auto axiom = app->callee()->isa<Axiom>()) {
                            if (axiom->tag() == Tag::Grad) {
                                return GradInfo(ds2cps->type()->as<Pi>(), lam, ds2cps->uses());
                            }
                        }
                    }
                }
            }
        }

        return {};
    }

}
