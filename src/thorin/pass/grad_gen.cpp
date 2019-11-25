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
    assert(!get_grad(var) && "We already have a gradient for this variable");
    def_to_partial_grads_.emplace(var, partial_grad);
}

const Def* GradGenEnv::sum_partial_grads(const Def* var) {
    (void)var;
    using DefPair = std::pair<const Def*, const Def*>;

    auto [begin, end] = def_to_partial_grads_.equal_range(var);
        if (begin == end) {
            return nullptr;
        }

        if (auto real_w = get_width(var->type())) {
        const Def* zero = world_.lit_real(*real_w, 0.0);
        auto add = world_.op(ROp::add);
        auto sum_up = [this, add](const Def* acc, DefPair cur) {
                            return world_.app(add, {acc, cur.second}); };
        auto sum = std::accumulate(begin, end, zero, sum_up);

        def_to_partial_grads_.erase(var);

        return sum;
    }

    // TODO: Show error
    THORIN_UNREACHABLE;
}

////////////////////////////////////////////////////////////////////////////////
// grad-gen pass
////////////////////////////////////////////////////////////////////////////////

const Def* GradGen::rewrite(const Def* def) {
    if (auto lam = has_lam_to_rewrite(def)) {
        auto grad_type = def->type()->as<Pi>();

        Array<const Def*> grads(grad_type->domain()->num_ops() - 2); // Minus mem and ret
        for (size_t i = 1; i < grad_type->domain()->num_ops() - 1; ++i) {
            grads[i - 1] = emit_grad(lam, lam->param(i, {}));
        }

        auto grad_lam = world().lam(grad_type, {});
        auto grad_ret = grad_lam->ret_param();
        auto grad_mem = grad_lam->mem_param();
        auto grad_tuple = world().tuple({grad_mem, world().tuple(grads)});
        auto grad_body = world().app(grad_ret, grad_tuple);

        grad_lam->set_body(grad_body);
        grad_lam->set_filter(world().lit_false());

        return grad_lam;
    }

    return nullptr;
}


const Def* GradGen::emit_grad(Lam* lam, const Def* var) {
    if (auto grad = env_.get_grad(var)) {
        return grad;
    }

    for (auto use : var->uses()) {
        for (auto op : uses_are_ops(use)) {
            auto j_wrapped = emit_J(op->as<App>());
            auto val = world().extract(j_wrapped, u64(0));
            auto B = world().extract(j_wrapped, u64(1));

            auto val_grad = emit_grad(lam, val);

            auto op_grads = world().app(B, val_grad);
            auto fst_op_param = world().extract(op->op(0), u64(0));
            auto snd_op_param = world().extract(op->op(0), u64(1));

            env_.add_partial_grad(fst_op_param, world().extract(op_grads, u64(0)));
            env_.add_partial_grad(snd_op_param, world().extract(op_grads, u64(1)));
        }

        if (use == lam->body()) {
            return world().lit_real(*get_width(var->type()), 1.0);
        }
    }

    return env_.sum_partial_grads(var);
}

const Def* GradGen::emit_J(const App* op) {
    return world().tuple({op, emit_pullback(op)});
}

const Def* GradGen::emit_pullback(const App* op) {
    auto op_mul = world().op(ROp::mul);
    auto op_div = world().op(ROp::div);

    auto axiom = op->callee()->as<Axiom>();
    auto real_t = op->type();
    auto real_w = *get_width(real_t);
    auto fst_op_param = world().extract(op->op(0), u64(0));
    auto snd_op_param = world().extract(op->op(0), u64(1));

    auto pullback_type = world().pi(real_t, world().sigma({real_t, real_t}));
    auto B = world().lam(pullback_type, {});
    auto param = B->param(B->num_params() - 1, {"∂f"});
    auto minus_param = world().app(op_mul, { world().lit_real(real_w, -1.0), param });
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
            auto fst_grad = world().app(op_mul, { param, snd_op_param });
            auto snd_grad = world().app(op_mul, { param, fst_op_param });

            B->set_body(world().tuple({ fst_grad, snd_grad }));
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
                        /// TODO: const cast seems wrong
                        return const_cast<Lam*>(cps2ds->cps()->as<Lam>());
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
                if (auto axiom = app->callee()->isa<Axiom>();
                         axiom && axiom->tag() == Tag::ROp) {
                    ops.push_back(app);
                }
            }
        }
    }

    return ops;
}


    ////////////////////////////////////////////////////////////////////////////////
    // grad-gen pass - old stuff
    ////////////////////////////////////////////////////////////////////////////////

    const Def* mk_add_pullback(World& world, const Def* real_type) {
        auto pullback_type = world.pi(real_type, world.sigma({real_type, real_type}));
        auto B = world.lam(pullback_type, {});
        auto param = B->param(B->num_params() - 1, {"∂z"});
        auto res = world.tuple({param, param});
        B->set_body(res);
        B->set_filter(world.lit_false());
        return B;
    }

    // TODO: does ADD
    const Def* mk_mul_pullback(World& world, const Def* real_type) {
        auto pullback_type = world.pi(real_type, world.sigma({real_type, real_type}));
        auto B = world.lam(pullback_type, {});
        auto param = B->param(B->num_params() - 1, {"∂z"});
        auto res = world.tuple({param, param});
        B->set_body(res);
        B->set_filter(world.lit_false());
        return B;
    }

    const Def* j_call(World& world, const Def* def) {
        if (auto outer = def->isa<App>()) {
            if (auto app = outer->op(0)->isa<App>()) {
                if (auto axiom = app->op(0)->isa<Axiom>()) {
                    switch (axiom->flags()) {
                    case (int)ROp::add: {
                        auto pullback = mk_add_pullback(world, outer->type());
                        auto wrapped = world.tuple({def, pullback});
                        return wrapped;
                    }
                    case (int)ROp::mul: {
                        auto pullback = mk_add_pullback(world, outer->type());
                        auto wrapped = world.tuple({def, pullback});
                        return wrapped;
                    }
                    }
                }
            }
        }

        return def;
    }

    const Def* GradGen::rewrite_old(const Def* def) {
        if (auto grad = isa_grad(def)) {
            auto ret_params = make_gradients(grad->lam);

            ////////// Head

            auto grad_lam = world().lam(grad->grad_type, {});
            auto grad_ret = grad_lam->ret_param();
            auto grad_mem = grad_lam->mem_param();
            //auto grad_domain = grad_lam->domain();

            ////////// Initialize gradients with one

            /*            auto grad_tangent_types = grad_domain->ops().skip_front().skip_back();
            Array<const Def*> param_tangents(grad_tangent_types.size());
            for (size_t i = 0; i < grad_tangent_types.size(); ++i) {
                param_tangents[i] = world().lit_tangent_one(grad_tangent_types[i]);
                }*/

            auto ret_val = world().tuple({grad_mem, ret_params});

            ////////// Call return continuation

            auto grad_body = world().app(grad_ret, ret_val);
            grad_lam->set_body(grad_body);
            grad_lam->set_filter(world().lit_false());

            errf("{}", grad_lam->body());
            //            return grad_lam;

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

    const Def* GradGen::make_gradients(const Lam* def) {
        auto pi = def->type();
        auto lam = const_cast<Lam*>(def); // :^)

        Array<const Def*> grads(pi->domain()->num_ops() - 2);
        // We start at the parameters and find where they are used. We search for functions.
        for (size_t i = 1; i < pi->domain()->ops().size() - 1; ++i) {
            for (auto& use : lam->param(i, {})->uses()) {
                // Stuff is wrapped in tuples before calling a function.
                if (use->type()->isa<Arr>()) {
                    for (auto inner_use : use->uses()) {
                        // Is this really a function?
                        if (auto app = inner_use->isa<App>()) {
                            auto j_called = j_call(world(), app);
                            auto new_app_val = world().extract(j_called, u64(0));
                            // Now replace the old direct uses of the return value with the J-wrapped return value
                            for (auto op_use : app->uses()) {
                                thorin::rewrite(op_use, app, new_app_val, Scope(lam));
                                // Find the return of the original function
                                for (auto res_use : op_use->uses()) {
                                    if (res_use == lam->body()) {
                                        auto B = world().extract(j_called, u64(1));
                                        auto delta_f = lam->body();
                                        auto deltas = world().app(B, delta_f);
                                        // Which of the deltas belongs to the current parameter?
                                        for (size_t idx = 0; idx < inner_use->op(1)->num_ops(); ++idx) {
                                            auto delta_param_i = world().extract(deltas, idx);
                                            grads[i - 1] = delta_param_i;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return world().tuple(grads);
    }

}
