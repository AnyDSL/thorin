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

        for (size_t i = 1; i < grad_type->domain()->num_ops() - 1; ++i) {
            grad_body = thorin::rewrite(grad_body, lam->param(i, {}), grad_lam->param(i, {}), Scope(lam));
        }

        grad_lam->set_body(grad_body);
        grad_lam->set_filter(world().lit_false());

        return grad_lam;
    }

    return def;
}


const Def* GradGen::emit_grad(Lam* lam, const Def* var) {
    if (auto grad = env_.get_grad(var)) {
        return grad;
    }

    for (auto use : var->copy_uses()) {
        for (auto op : uses_are_ops(use)) {
            auto j_wrapped = emit_J(op->as<App>());
            auto val = world().extract(j_wrapped, u64(0));
            thorin::rewrite(lam, use, val, Scope(lam));

            auto val_grad = emit_grad(lam, val);

            auto B = world().extract(j_wrapped, u64(1));
            auto op_grads = world().app(B, val_grad);

            auto fst_op_param = world().extract(op->op(1), u64(0));
            env_.add_partial_grad(fst_op_param, world().extract(op_grads, u64(0)));

            auto snd_op_param = world().extract(op->op(1), u64(1));
            env_.add_partial_grad(snd_op_param, world().extract(op_grads, u64(1)));
        }

        if (use_is_ret(lam, use)) {
            return world().lit_real(*get_width(var->type()), 1.0);
        }
    }

    if (auto grad = env_.sum_partial_grads(var)) {
        return grad;
    } else {
        THORIN_BREAK;
        return nullptr;
    }
}

const Def* GradGen::emit_J(const App* op) {
    return world().tuple({op, emit_pullback(op)});
}

const Def* GradGen::emit_pullback(const App* op) {
    auto op_mul = world().op(ROp::mul);
    auto op_div = world().op(ROp::div);

    auto axiom = op->callee()->as<App>()->callee()->as<Axiom>();
    auto real_t = op->type();
    auto real_w = *get_width(real_t);
    auto fst_op_param = world().extract(op->op(1), u64(0));
    auto snd_op_param = world().extract(op->op(1), u64(1));

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

