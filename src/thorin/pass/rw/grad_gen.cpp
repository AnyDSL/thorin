#include "thorin/pass/rw/grad_gen.h"

#include "thorin/rewrite.h"

#include <numeric>

namespace thorin {

GradEmitter::GradEmitter(Lam* orig_lam, Lam* grad_lam)
    : orig_lam_(orig_lam)
    , grad_lam_(grad_lam)
    , orig_scope_(Scope(orig_lam))
    , world_(orig_lam->world())
    , rewriter_(world_, &orig_scope_)
    , pullback_gens_ {{
        [size_t(ROp::add)] = [this](auto op){ return pullback_for_add(op); },
        [size_t(ROp::sub)] = [this](auto op){ return pullback_for_sub(op); },
        [size_t(ROp::mul)] = [this](auto op){ return pullback_for_mul(op); },
        [size_t(ROp::div)] = [this](auto op){ return pullback_for_div(op); },
        [size_t(ROp::mod)] = nullptr,
    }}
{
    check_initial_sanity();

    for (size_t i = 0; i < num_params(); ++i) {
        rewriter_.old2new[orig_param(i)] = grad_param(i);
    }
}

Lam* GradEmitter::emit_grad_lam() {
    fill_grads_for_orig_params();
    set_grad_lam_body();

    return grad_lam_;
}

void GradEmitter::fill_grads_for_orig_params() {
    for (size_t i = 0; i < num_params(); ++i) {
        auto param = orig_param(i);

        if (param->num_uses() > 0) {
            var_to_grads_[param] = emit_grad_for_var(param);
        } else {
            errf("warning: grad-gen: {}. parameter of {} is unused", i+1, grad_lam_->debug().name);
            var_to_grads_[param] = world_.bot(param->type());
        }
    }
}

void GradEmitter::set_grad_lam_body() {
    Array<const Def*> grads(num_params());

    for (size_t i = 0; i < num_params(); ++i) {
        auto param = orig_param(i);

        if (auto grad = var_to_grads_[param]) {
            grads[i] = grad;
        } else {
            errf("warning: grad-gen: Failed to gen grad for {}. param of {}", i+1, grad_lam_->debug().name);
            grads[i] = world_.bot(param->type());
        }
    }

    auto ret  = grad_lam_->ret_param(world_.dbg("return"));
    auto mem  = grad_lam_->mem_param(world_.dbg("mem"));
    auto body = world_.app(ret, world_.tuple({ mem, world_.tuple(grads) }, world_.dbg("result")));

    grad_lam_->set_body(body);
    grad_lam_->set_filter(world_.lit_false());
}

const Def* GradEmitter::orig_param(size_t i) const {
    return orig_lam_->param(i + 1);
}

const Def* GradEmitter::grad_param(size_t i) const {
    return grad_lam_->param(i + 1, orig_param(i)->dbg() );
}

const Def* GradEmitter::emit_grad_for_var(const Def* var) {
    for (auto use : var->copy_uses()) {
        if (orig_scope_.contains(use)) {
            if (emit_partial_grad_for_rop_use(var, use)) {
                // TODO: I am pretty sure we want to use the result for something...
            } else if (emit_partial_grad_for_ret_use(var, use)) {
                // TODO: I am pretty sure we want to use the result for something...
            } else {
                errf("!error: grad-gen: operation {} is not supported for {}", use, var);
                return world_.bot(var->type());
            }
        }
    }

    return var_to_grads_[var];
}

const Def* GradEmitter::emit_partial_grad_for_rop_use(const Def* var, const Def* use) {
    if (use->type()->isa<Arr>()) {
        for (auto useuse : use->uses()) {
            if (auto app = useuse->isa<App>()) {
                if (auto maybe_axiom_app = app->callee()->isa<App>()) {
                    if (auto axiom = maybe_axiom_app->callee()->isa<Axiom>();
                        axiom && axiom->tag() == Tag::ROp) {

                        auto B = emit_pullback_for_rop(app);
                        auto op_grad = visited_.contains(app) ? var_to_grads_[app]
                                                              : emit_grad_for_var(app);

                        if (!op_grad) {
                            errf("Could not get gradient for {}", app);
                            continue;
                        }

                        visited_.emplace(app);
                        add_partial_grads_for_rop(var, app, B, op_grad);
                    }
                }
            }
        }
    }

    return var_to_grads_[var];
}

const Def* GradEmitter::emit_partial_grad_for_ret_use(const Def* var, const Def* use) {
    if (use->isa<Tuple>()) {
        for (auto useuse : use->uses()) {
            if (useuse == orig_lam_->body()) {
                var_to_grads_[var] = world_.lit_real(as_lit(isa_sized_type(var->type())), u64(1.0));
                return var_to_grads_[var];
            }
        }
    }

    return nullptr;
}

const Def* GradEmitter::into_grad_scope(const Def* old_def) {
    if (auto new_def = rewriter_.old2new.lookup(old_def)) return *new_def;
    if (old_def->isa<Param>()) return old_def;

    for (auto op : old_def->ops()) {
        rewriter_.old2new[op] = into_grad_scope(op);
    }

    return rewriter_.rewrite(old_def);
}

const Def* GradEmitter::emit_pullback_for_rop(const Def* op) {
    op = into_grad_scope(op);

    try {
        auto axiom = op->as<App>()->callee()->as<App>()->callee()->as<Axiom>();
        if (auto gen = pullback_gens_.at(axiom->flags())) {
            if (!use_to_pullbacks_.contains(op)) {
                use_to_pullbacks_[op] = gen(op);
            }
            return use_to_pullbacks_[op] ;
        }
    } catch(std::out_of_range&) {}

    errf("error: grad-gen cannot create a pullback for {}", op);
    return world_.bot(op->type());
}

// ∇(a + b) = λ∂f.[∂f, ∂f]
const Def* GradEmitter::pullback_for_add(const Def* op) {
    auto [fst_op, snd_op] = op->as<App>()->args<2>();
    // TODO same below

    auto pi = world_.pi(op->type(), world_.sigma({ fst_op->type(), snd_op->type() }));
    auto B = world_.nom_lam(pi, world_.dbg("B⁺"));
    auto param = B->param(world_.dbg("∂f"));

    B->set_filter(world_.lit_false());
    B->set_body(world_.tuple({ param, param }));
    return B;
}

// ∇(a - b) = λ∂f.[∂f, -∂f]
const Def* GradEmitter::pullback_for_sub(const Def* op) {
    auto fst_op = world_.extract(op->op(1), 2, 0_s, world_.dbg("op₀"));
    auto snd_op = world_.extract(op->op(1), 2, 1_s, world_.dbg("op₁"));

    auto pi = world_.pi(op->type(), world_.sigma({ fst_op->type(), snd_op->type() }));
    auto B = world_.nom_lam(pi, world_.dbg("B⁻"));
    auto param = B->param(world_.dbg("∂f"));
    auto param_w = as_lit(isa_sized_type(param->type()));
    auto mul = world_.app(world_.op(ROp::mul), { world_.lit_nat(param_w), world_.lit_nat(param_w) });
    auto neg_param = world_.app(mul, { world_.lit_real(param_w, -1.0), param }, world_.dbg("∂ɟ"));

    B->set_filter(world_.lit_false());
    B->set_body(world_.tuple({ param, neg_param }));
    return B;
}

// ∇(a * b) = λ∂f.[∂f*b, ∂f*a]
const Def* GradEmitter::pullback_for_mul(const Def* op) {
    auto fst_op = world_.extract(op->op(1), 2, 0_s, world_.dbg("op₀"));
    auto fst_w = world_.lit_nat(as_lit(isa_sized_type(fst_op->type())));
    auto snd_op = world_.extract(op->op(1), 2, 1_s, world_.dbg("op₁"));
    auto snd_w = world_.lit_nat(as_lit(isa_sized_type(snd_op->type())));

    auto pi = world_.pi(op->type(), world_.sigma({ fst_op->type(), snd_op->type() }));
    auto B = world_.nom_lam(pi, world_.dbg("B×"));
    auto param = B->param(world_.dbg("∂f"));
    auto param_w = world_.lit_nat(as_lit(isa_sized_type(param->type())));

    auto fst_mul = world_.app(world_.op(ROp::mul), { param_w, snd_w });
    auto fst_grad = world_.app(fst_mul, { param, snd_op }, world_.dbg("∂" + fst_op->debug().name));
    auto snd_mul = world_.app(world_.op(ROp::mul), { param_w, fst_w });
    auto snd_grad = world_.app(snd_mul, { param, fst_op }, world_.dbg("∂" + snd_op->debug().name));

    B->set_filter(world_.lit_false());
    B->set_body(world_.tuple({ fst_grad, snd_grad }));
    return B;
}

// ∇(a / b) = λ∂f.[∂f/b, (-∂f*a)/(b²)]
const Def* GradEmitter::pullback_for_div(const Def* op) {
    auto fst_op = world_.extract(op->op(1), 2, 0_s, world_.dbg("op₀"));
    auto fst_w = world_.lit_nat(as_lit(isa_sized_type(fst_op->type())));
    auto snd_op = world_.extract(op->op(1), 2, 1_s, world_.dbg("op₁"));
    auto snd_w = world_.lit_nat(as_lit(isa_sized_type(snd_op->type())));

    auto pi = world_.pi(op->type(), world_.sigma({ fst_op->type(), snd_op->type() }));
    auto B = world_.nom_lam(pi, world_.dbg("B÷"));
    auto param = B->param(world_.dbg("∂f"));
    auto param_w = as_lit(isa_sized_type(param->type()));
    auto neg_mul = world_.app(world_.op(ROp::mul), { world_.lit_nat(param_w), world_.lit_nat(param_w) });
    auto neg_param = world_.app(neg_mul, { world_.lit_real(param_w, -1.0), param }, world_.dbg("∂ɟ"));

    auto fst_div = world_.app(world_.op(ROp::div), { world_.lit_nat(param_w), snd_w });
    auto fst_grad = world_.app(fst_div, { param, snd_op }, world_.dbg("∂" + fst_op->debug().name));
    auto snd_up_mul = world_.app(world_.op(ROp::mul), { world_.lit_nat(param_w), fst_w });
    auto snd_up_grad = world_.app(snd_up_mul, { neg_param, fst_op }, world_.dbg("∂" + snd_op->debug().name + "₀"));
    auto snd_low_mul = world_.app(world_.op(ROp::mul), { snd_w, snd_w });
    auto snd_low_grad = world_.app(snd_low_mul, { snd_op, snd_op }, world_.dbg("∂" + snd_op->debug().name + "₁"));
    auto snd_up_w = world_.lit_nat(as_lit(isa_sized_type(snd_up_grad->type())));
    auto snd_low_w = world_.lit_nat(as_lit(isa_sized_type(snd_low_grad->type())));
    auto snd_div = world_.app(world_.op(ROp::div), { snd_up_w, snd_low_w });
    auto snd_grad = world_.app(snd_div, { snd_up_grad, snd_low_grad }, world_.dbg("∂" + snd_op->debug().name));

    B->set_filter(world_.lit_false());
    B->set_body(world_.tuple({ fst_grad, snd_grad }));
    return B;
}

void GradEmitter::add_partial_grads_for_rop(const Def* var, const Def* op, const Def* pullback, const Def* op_grad) {
    auto grads = world_.app(pullback, op_grad, world_.dbg("∇op" + op->debug().name));

    auto add_part_grad_for_param =
        [this, var, op, grads](size_t i) {
            auto param = world_.extract(op->op(1), 2, i);                                       // TODO 2 arity correct?
            if (var == param) {
                auto grad = world_.extract(grads, 2, i, world_.dbg("∂" + param->debug().name)); // TODO 2 arity correct?
                add_partial_grad(var, grad);
            }
        };

    add_part_grad_for_param(0);
    add_part_grad_for_param(1);
}

void GradEmitter::add_partial_grad(const Def* var, const Def* part_grad) {
    auto iter = var_to_grads_.find(var);

    if (iter != var_to_grads_.end()) {
        auto part_grad_w = world_.lit_nat(as_lit(isa_sized_type(part_grad->type())));
        auto grad_so_far_w = world_.lit_nat(as_lit(isa_sized_type(iter->second->type())));
        auto add = world_.app(world_.op(ROp::add), { part_grad_w, grad_so_far_w });

        var_to_grads_[var] = world_.app(add, { part_grad, iter->second }, world_.dbg("∂" + var->debug().name));
    } else {
        var_to_grads_[var] = part_grad;
    }
}

const Def* GradGen::rewrite(Def*, const Def* def) {
    if (auto lam = has_lam_to_rewrite(def)) {

        auto grad_type = def->type()->as<Pi>();
        auto grad_lam = world().nom_lam(grad_type, world().dbg("∇" + lam->debug().name));

        GradEmitter emitter(lam, grad_lam);
        return emitter.emit_grad_lam();
    }

    return def;
}

Lam* GradGen::has_lam_to_rewrite(const Def* /*def*/) const {
#if 0
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
#endif
    return nullptr;
}

}
