#include <thorin/pass/grad_gen.h>
#include <thorin/rewrite.h>
#include <thorin/util.h>

#include <numeric>

namespace thorin {

////////////////////////////////////////////////////////////////////////////////
// environment
////////////////////////////////////////////////////////////////////////////////

const Def* GradGenEnv::get_grad(const Def* var) {
    return def_to_grads_[var];
}

void GradGenEnv::add_partial_grad(const Def* var, const  Def* partial_grad) {
    if (auto w = get_width(var->type())) {
        auto iter = def_to_grads_.find(var);

        if (iter == def_to_grads_.end() || !iter->second) {
            def_to_grads_[var] = partial_grad;
            return;
        }

        auto w_lit = world_.lit_nat(*w);
        auto type_args = world_.tuple({ w_lit, w_lit });
        auto add = world_.app(world_.op(ROp::add), type_args);
        auto new_part_grad = world_.app(add, { iter->second, partial_grad }, { "∂" + var->name() });

        def_to_grads_[var] = new_part_grad;
        return;
    }

    assert(false && "Only gradients of reals are supported");
}

////////////////////////////////////////////////////////////////////////////////
// grad-gen pass
////////////////////////////////////////////////////////////////////////////////

const Def* GradGen::rewrite(const Def* def) {
    if (auto lam = has_lam_to_rewrite(def)) {
        auto grad_type = def->type()->as<Pi>();
        auto grad_lam = world().lam(grad_type, {"∇" + lam->name()});

        Array<const Def*> grads(grad_type->domain()->num_ops() - 2); // Minus mem and ret
        for (size_t i = 1; i < grad_type->domain()->num_ops() - 1; ++i) {
            grads[i - 1] = emit_grad(lam, grad_lam, lam->param(i));
        }

        auto grad_ret = grad_lam->ret_param({"return"});
        auto grad_mem = grad_lam->mem_param({"mem"});
        auto grad_tuple = world().tuple({grad_mem, world().tuple(grads)});
        auto grad_body = world().app(grad_ret, grad_tuple);

        for (size_t i = 1; i < grad_type->domain()->num_ops() - 1; ++i) {
            auto lam_param = lam->param(i);
            auto grad_param = lam->param(i, { lam_param->name() });
            grad_body = thorin::rewrite(grad_body, lam_param, grad_param, Scope(lam));
        }

        grad_lam->set_body(grad_body);
        grad_lam->set_filter(world().lit_false());

        Scope(lam).dump();
        errf("goes to\n");
        Scope(grad_lam).dump();

        return grad_lam;
    }

    return def;
}

const Def* GradGen::emit_grad(Lam* lam, Lam* grad_lam, const Def* var) {
    if (auto grad = env_.get_grad(var)) {
        return grad;
    }

    for (auto use : var->copy_uses()) {
        for (auto op : uses_are_ops(use)) {
            auto j_wrapped = emit_J(op->as<App>());
            for (size_t i = 1; i < grad_lam->domain()->num_ops() - 1; ++i) {
                auto lam_param = lam->param(i);
                auto grad_param = grad_lam->param(i, {lam_param->name()}) ;
                j_wrapped = thorin::rewrite(j_wrapped, lam_param, grad_param, Scope(lam));
            }

            if (auto val_grad = emit_grad(lam, grad_lam, op)) {
                auto B = world().extract(j_wrapped, u64(1));
                auto op_grads = world().app(B, val_grad, {"∇ops"});

                auto fst_op_param = world().extract(op->op(1), u64(0));
                if (var == fst_op_param) {
                    auto part_grad = world().extract(op_grads, u64(0), {"∂" + fst_op_param->name()});
                    env_.add_partial_grad(fst_op_param, part_grad);
                }

                auto snd_op_param = world().extract(op->op(1), u64(1));
                if (var == snd_op_param) {
                    auto part_grad = world().extract(op_grads, u64(1), {"∂" + fst_op_param->name()});
                    env_.add_partial_grad(snd_op_param, part_grad);
                }
            }
            else {
                return world().lit_real(64, r64(1.0), {"one"});
            }
        }

        if (use_is_ret(lam, use)) {
            return world().lit_real(64, r64(1.0), {"one"});
        }
    }

    if (auto grad =  env_.get_grad(var)) {
        return grad;
    } else {
        return nullptr;
    }
}

const Def* GradGen::emit_J(const App* op) {
    return world().tuple({op, emit_pullback(op)}, {"J"});
}

const Def* GradGen::emit_pullback(const App* op) {
    auto axiom = op->callee()->as<App>()->callee()->as<Axiom>();
    auto real_t = world().type_real(64);
    //auto real_w = 64;// *get_width(real_t);
    auto fst_op_param = world().extract(op->op(1), u64(0), {"op0"});
    auto snd_op_param = world().extract(op->op(1), u64(1), {"op1"});

    //auto type_args = world().tuple({ fst_op_param->type(), snd_op_param->type() });
    auto type_args = world().tuple({ world().lit_nat(64), world().lit_nat(64) });
    auto op_mul = world().app(world().op(ROp::mul), type_args);
    auto op_div = world().app(world().op(ROp::div), type_args);

    auto pullback_type = world().pi(real_t, world().sigma({real_t, real_t}));
    auto B = world().lam(pullback_type, {"B"});
    auto param = B->param({"∂f"});
    auto minus_param = world().app(op_mul, { world().lit_real(64, r64(-1.0), {"minus_one"}), param });
    B->set_filter(world().lit_false());

    switch (axiom->flags()) {
        /// ∇(a + b) = λ∂f.[∂f, ∂f]
        case (int)ROp::add: {
            B->set_body(world().tuple({ param, param}));
            return B;
        }
        /// ∇(a - b) = λ∂f.[∂f, -∂f]
        /// TODO: is that correct?
        case (int)ROp::sub: {
            B->set_body(world().tuple({ param, minus_param }));
            return B;
        }
        /// ∇(a * b) = λ∂f.[∂f*b, ∂f*a]
        case (int)ROp::mul: {
            auto fst_grad = world().app(op_mul, { param, snd_op_param }, {"∂" + fst_op_param->name()});
            auto snd_grad = world().app(op_mul, { param, fst_op_param }, {"∂" + snd_op_param->name()});

            B->set_body(world().tuple({ fst_grad, snd_grad }, {}));
            return B;
        }
        /// ∇(a / b) = λ∂f.[∂f/b, (-∂f*a)/(b²)]
        case (int)ROp::div: {
            auto fst_grad = world().app(op_div, { param, fst_op_param });
            auto snd_grad = world().app(op_div, { world().app(op_mul, { minus_param, fst_op_param }),
                                                  world().app(op_mul, { snd_op_param, snd_op_param }) });

            B->set_body(world().tuple({ fst_grad, snd_grad }));
            return B;
        }
    }

    THORIN_UNREACHABLE;
}

Lam* GradGen::has_lam_to_rewrite(const Def* def) const {
    if (auto ds2cps = def->isa<DS2CPS>()) {
        if (auto app_to_grad = ds2cps->ds()->isa<App>();
                 app_to_grad && app_to_grad->num_args() > 0) {
            if (auto app_grad = app_to_grad->callee()->isa<App>()) {
                if (auto axiom = app_grad->callee()->isa<Axiom>();
                    axiom && axiom->tag() == Tag::Grad) {
                    if (auto cps2ds = app_to_grad->arg(0)->isa<CPS2DS>()) {
                        return cps2ds->cps()->isa_nominal<Lam>();
                    }
                }
            }
        }
    }

    return nullptr;
}

std::vector<const Def*> GradGen::uses_are_ops(const Def* use) const {
    std::vector<const Def*> ops;

    if (use->type()->isa<Arr>()) {
        for (auto use_of_tuple : use->uses()) {
            if (auto app = use_of_tuple->isa<App>()) {
                if (auto maybe_axiom_app = app->callee()->isa<App>()) {
                    if (auto axiom = maybe_axiom_app->callee()->isa<Axiom>();
                        axiom && axiom->tag() == Tag::ROp) {
                        ops.push_back(app);
                    }
                }
            }
        }
    }

    return ops;
}

bool GradGen::use_is_ret(Lam* lam, const Def* use) const {
    if (auto tuple = use->isa<Tuple>()) {
        for (auto tuple_use : tuple->uses()) {
            if (use_is_ret(lam, tuple_use)) {
                return true;
            }
        }
    }

    if (use == lam->body()) {
        return true;
    }

    return false;
}

} //namespace thorin

