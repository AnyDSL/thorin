#include "thorin/pass/rw/auto_diff.h"

#include <algorithm>
#include <string>

#include "thorin/analyses/scope.h"

namespace thorin {

namespace {

class AutoDiffer {
public:
    AutoDiffer(World& world, const Def* gradient, const Def2Def src_to_dst)
        : world_{world}
        , gradient_{gradient}
        , src_to_dst_{src_to_dst} {}

    const Def* reverse_diff(const Def* src);
    const Def* forward_diff(const Def*) { throw "not implemented"; }

    const Def* grad(const Def* def);

private:
    const Def* j_wrap(const Def* def);
    const Def* j_wrap_rop(ROp op, const Def* a, const Def* b);

    void fill_grad(const Def* def, const Def* cur_grad);

    const Def* seen(const Def* src);
    const Def* add_part_grad(const Def* primal_def, const Def* part_grad);

    World& world_;
    const Def* gradient_;
    Def2Def src_to_dst_;
    Def2Def dst_to_pullback_;
    Def2Def dst_to_part_diff_;
};

const Def* AutoDiffer::reverse_diff(const Def* src) {
    auto dst = j_wrap(src);
    fill_grad(dst, gradient_);
    return dst;
}

const Def* AutoDiffer::j_wrap(const Def* def) {
    if (auto dst = seen(def))
        return dst;

    if (auto var = def->isa<Var>()) {
        errf("This var must be out of scope: {}\n This is not differentiable", var);
        THORIN_UNREACHABLE;
    }

    if (auto axiom = def->isa<Axiom>()) {
        errf("Found a non-differentiable axiom: {}", axiom);
        THORIN_UNREACHABLE;
    }

    if (auto lam = def->isa_nom<Lam>()) {
        auto dst = world_.nom_lam(lam->type(), lam->dbg());
        src_to_dst_[lam->var()] = dst->var();
        dst->set_filter(lam->filter());
        dst->set_body(j_wrap(lam->body()));
        src_to_dst_[lam] = dst;
        return dst;
    }

    if (auto app = def->isa<App>()) {
        auto callee = app->callee();
        auto arg = app->arg();

        if (auto inner = callee->isa<App>()) {
            if (auto axiom = inner->callee()->isa<Axiom>()) {
                if (axiom->tag() == Tag::ROp) {
                    auto [a, b] = j_wrap(arg)->split<2>();
                    auto [dst, pb] = j_wrap_rop(ROp(axiom->flags()), a, b)->split<2>();
                    dst_to_pullback_[dst] = pb;
                    src_to_dst_[app] = dst;
                    return dst;
                }

                if (axiom->tag() == Tag::RCmp) {
                    auto [a, b] = j_wrap(arg)->split<2>();
                    auto dst = world_.op(RCmp(axiom->flags()), nat_t(0), a, b);
                    src_to_dst_[app] = dst;
                    return dst;
                }
            }
        }

        if (callee->type()->as<Pi>()->is_returning()) {
            auto rd_callee = world_.op_rev_diff(callee);

            auto wrapper_pi = rd_callee->type()->as<Pi>()->doms().back()->as<Pi>();
            auto wrapper_lam = world_.nom_lam(wrapper_pi, world_.dbg("wrapper"));

            wrapper_lam->set_filter(world_.lit_true());
            wrapper_lam->set_body(world_.app(j_wrap(app->args().back()), wrapper_lam->vars().skip_back()));

            auto num_args = app->num_args();
            Array<const Def*> args{
                num_args, [&](auto i) { return i < num_args - 1 ? j_wrap(app->arg(i)) : wrapper_lam; }};

            auto dst = world_.app(rd_callee, args);
            dst_to_pullback_[dst] = wrapper_lam->vars().back();
            src_to_dst_[app] = dst;
            return dst;
        }

        auto dst = world_.app(j_wrap(callee), j_wrap(arg));
        src_to_dst_[app] = dst;
        return dst;
    }

    if (auto tuple = def->isa<Tuple>()) {
        Array<const Def*> ops{tuple->num_ops(), [&](auto i) { return j_wrap(tuple->op(i)); }};
        auto dst = world_.tuple(ops);
        src_to_dst_[tuple] = dst;
        return dst;
    }

    if (auto pack = def->isa<Pack>()) {
        auto dst = world_.pack(pack->type()->arity(), j_wrap(pack->body()));
        src_to_dst_[pack] = dst;
        return dst;
    }

    if (auto extract = def->isa<Extract>()) {
        auto dst = world_.extract(j_wrap(extract->tuple()), j_wrap(extract->index()));
        src_to_dst_[extract] = dst;
        return dst;
    }

    if (auto insert = def->isa<Insert>()) {
        auto dst = world_.insert(j_wrap(insert->tuple()), j_wrap(insert->index()), j_wrap(insert->value()));
        src_to_dst_[insert] = dst;
        return dst;
    }

    if (auto lit = def->isa<Lit>()) {
        return lit;
    }

    errf("I don't know yet how to handle: {}", def);
    THORIN_UNREACHABLE;
}

const Def* AutoDiffer::j_wrap_rop(ROp op, const Def* a, const Def* b) {
    auto r_type = a->type();
    auto pi = world_.pi(r_type, world_.sigma({r_type, r_type}));

    switch (op) {
        // ∇(a + b) = λ∂f.[∂f, ∂f]
        case ROp::add: {
            auto B = world_.nom_lam(pi, world_.dbg("φ+"));
            auto var = B->var();
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({var, var}));
            return world_.tuple({world_.op(ROp::add, (nat_t)0, a, b), B});
        }
        // ∇(a - b) = λ∂f.[∂f, -∂f]
        case ROp::sub: {
            auto B = world_.nom_lam(pi, world_.dbg("φ-"));
            auto var = B->var();
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({var, world_.op_rminus((nat_t)0, var)}));
            return world_.tuple({world_.op(ROp::sub, (nat_t)0, a, b), B});
        }
        // ∇(a * b) = λ∂f.[∂f*b, ∂f*a]
        case ROp::mul: {
            auto B = world_.nom_lam(pi, world_.dbg("φ*"));
            auto var = B->var();
            auto d1 = world_.op(ROp::mul, nat_t(0), var, b);
            auto d2 = world_.op(ROp::mul, nat_t(0), var, a);
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({d1, d2}));
            return world_.tuple({world_.op(ROp::mul, (nat_t)0, a, b), B});
        }
        // ∇(a / b) = λ∂f.[∂f/b, (-∂f*a)/(b²)]
        case ROp::div: {
            auto B = world_.nom_lam(pi, world_.dbg("φ/"));
            auto var = B->var();
            auto neg_var = world_.op_rminus(nat_t(0), B->var());
            auto d1 = world_.op(ROp::div, nat_t(0), var, b);
            auto numerator = world_.op(ROp::mul, nat_t(0), neg_var, a);
            auto denominator = world_.op(ROp::mul, nat_t(0), b, b);
            auto d2 = world_.op(ROp::div, nat_t(0), numerator, denominator);
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({d1, d2}));
            return world_.tuple({world_.op(ROp::div, (nat_t)0, a, b), B});
        }
        default:
            THORIN_UNREACHABLE;
    }
}

void AutoDiffer::fill_grad(const Def* def, const Def* cur_grad) {
    if (auto app = def->isa<App>()) {
        if (dst_to_pullback_.contains(app)) {
            auto pb = dst_to_pullback_[app];

            if (pb->type()->as<Pi>()->is_cn()) {
                /*
                auto lam = world_.nom_lam(pb->type()->as<Pi>(), world_.dbg("adjoint_cn"));
                auto grads = lam->vars().skip_back().skip_back();
                for (size_t i = 0; i < app->num_args(); ++i) {
                    fill_grad(app->arg(i), grads[i]);
                }
                */
            } else {
                auto grads = world_.app(pb, cur_grad, world_.dbg("∇" + app->callee()->name()));
                for (size_t i = 0; i < app->num_args(); ++i) {
                    fill_grad(app->arg(i), world_.extract(grads, i, world_.dbg("∇" + std::to_string(i))));
                }
            }
        } else {
            for (size_t i = 0; i < app->num_args(); ++i) {
                fill_grad(app->arg(i), cur_grad);
            }
        }
    }

    if (auto tuple = def->isa<Tuple>()) {
        for (auto op : tuple->ops()) {
            fill_grad(op, cur_grad);
        }
    }

    if (auto pack = def->isa<Pack>()) {
        fill_grad(pack->body(), cur_grad);
    }

    if (auto extract = def->isa<Extract>()) {
        if (/*auto var = */ extract->tuple()->isa<Var>()) { // TODO this looks broken?
            add_part_grad(extract, cur_grad);
        } else {
            fill_grad(extract->tuple(), cur_grad);
        }
    }

    if (def->isa<Insert>()) {
        errf("Insert is not supported");
        THORIN_UNREACHABLE;
    }
}

const Def* AutoDiffer::grad(const Def* def) {
    if (!dst_to_part_diff_.contains(def)) {
        return world_.lit_real(r64(0));
    }
    return dst_to_part_diff_[def];
}

const Def* AutoDiffer::add_part_grad(const Def* primal_def, const Def* part_grad) {
    if (!dst_to_part_diff_.contains(primal_def)) {
        dst_to_part_diff_[primal_def] = part_grad;
        return part_grad;
    }

    auto old_part_grad = dst_to_part_diff_[primal_def];
    auto new_part_grad = world_.op(ROp::add, nat_t(0), old_part_grad, part_grad, world_.dbg("∂" + part_grad->name()));
    dst_to_part_diff_[primal_def] = new_part_grad;

    return new_part_grad;
}

const Def* AutoDiffer::seen(const Def* def) { return src_to_dst_.contains(def) ? src_to_dst_[def] : nullptr; }

} // namespace

const Def* AutoDiff::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto type_app = app->callee()->isa<App>()) {
            if (auto axiom = type_app->callee()->isa<Axiom>(); axiom && axiom->tag() == Tag::RevDiff) {
                auto src_lam = app->arg(0)->as_nom<Lam>();
                auto src_pi = src_lam->type()->as<Pi>();
                auto& world = src_lam->world();

                // Copy of the original function, augmented to return <r32, f(32)->r32>
                auto dst_pi = app->type()->as<Pi>();
                auto dst_lam = world.nom_lam(dst_pi, world.dbg("rev_diff_" + src_lam->name()));

                auto pb_pi = dst_pi->doms().back()->as<Pi>();
                auto pb_lam = world.nom_lam(pb_pi, world.dbg("pullback_" + src_lam->name()));
                auto [pb_mem, pb_grad, pb_ret] = pb_lam->var()->split<3>();

                auto ret_pi = src_pi->doms().back()->as<Pi>();
                auto ret_lam = world.nom_lam(ret_pi, world.dbg("wrapper"));

                Def2Def src_to_dst;
                for (size_t i = 0, e = src_lam->num_vars(); i < e; ++i) {
                    auto src_var = src_lam->var(i);
                    auto dst_var = dst_lam->var(i, world.dbg(src_var->name()));
                    src_to_dst[src_var] = i == e - 1 ? ret_lam : dst_var;
                }
                auto differ = AutoDiffer{world, pb_grad, src_to_dst};

                dst_lam->set_filter(src_lam->filter());
                dst_lam->set_body(differ.reverse_diff(src_lam->body()));

                auto num_ret_body_args = dst_pi->doms().back()->as<Pi>()->num_doms();
                Array<const Def*> ret_body_args{num_ret_body_args,
                    [&](auto i) { return i < num_ret_body_args - 1 ? ret_lam->var(i) : pb_lam; }};
                ret_lam->set_filter(world.lit_true());
                ret_lam->set_body(world.app(dst_lam/*->ret_var(world.dbg("return"))*/, ret_body_args));

                // TODO: what if multidimensional output?
                auto num_grads = src_lam->num_vars() - 2;
                Array<const Def*> grads{num_grads, [&](auto i) { return differ.grad(dst_lam->var(i + 1)); }};
                pb_lam->set_filter(world.lit_true());
                pb_lam->set_body(world.app(pb_lam->ret_var(world.dbg("pb_ret")), {pb_lam->mem_var(), world.tuple(grads), ret_lam}));

                return dst_lam;
            }
        }
    }

    return def;
}

}
