#include "auto_diff.h"

#include "../analyses/scope.h"
#include <algorithm>
#include <string>

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
    THORIN_BREAK;

    if (auto dst = seen(def))
        return dst;

    if (auto param = def->isa<Param>()) {
        errf("This param must be out of scope: {}\n This is not differentiable", param);
        THORIN_UNREACHABLE;
    }

    if (auto axiom = def->isa<Axiom>()) {
        errf("Found a non-differentiable axiom: {}", axiom);
        THORIN_UNREACHABLE;
    }

    if (auto lam = def->isa_nominal<Lam>()) {
    }

    if (auto app = def->isa<App>()) {
        if (auto inner = app->callee()->isa<App>()) {
            if (auto axiom = inner->callee()->isa<Axiom>()) {
                if (axiom->tag() == Tag::ROp) {
                    auto [a, b] = j_wrap(app->arg())->split<2>();
                    auto [dst, pb] = j_wrap_rop(ROp(axiom->flags()), a, b)->split<2>();
                    dst_to_pullback_[dst] = pb;
                    src_to_dst_[app] = dst;
                    return dst;
                }

                if (axiom->tag() == Tag::RCmp) {
                    auto [a, b] = j_wrap(app->arg())->split<2>();
                    auto dst = world_.op(RCmp(axiom->flags()), nat_t(0), a, b);
                    src_to_dst_[app] = dst;
                    return dst;
                }
            }
        }

        auto dst = world_.app(j_wrap(app->callee()), j_wrap(app->arg()));
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
            auto B = world_.lam(pi, {"φ+"});
            auto param = B->param();
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({param, param}));
            return world_.tuple({world_.op(ROp::add, (nat_t)0, a, b), B});
        }
        // ∇(a - b) = λ∂f.[∂f, -∂f]
        case ROp::sub: {
            auto B = world_.lam(pi, {"φ-"});
            auto param = B->param();
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({param, world_.op_ROp_minus((nat_t)0, param)}));
            return world_.tuple({world_.op(ROp::sub, (nat_t)0, a, b), B});
        }
        // ∇(a * b) = λ∂f.[∂f*b, ∂f*a]
        case ROp::mul: {
            auto B = world_.lam(pi, {"φ*"});
            auto param = B->param();
            auto d1 = world_.op(ROp::mul, nat_t(0), param, b);
            auto d2 = world_.op(ROp::mul, nat_t(0), param, a);
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({d1, d2}));
            return world_.tuple({world_.op(ROp::mul, (nat_t)0, a, b), B});
        }
        // ∇(a / b) = λ∂f.[∂f/b, (-∂f*a)/(b²)]
        case ROp::div: {
            auto B = world_.lam(pi, {"φ/"});
            auto param = B->param();
            auto neg_param = world_.op_ROp_minus(nat_t(0), B->param());
            auto d1 = world_.op(ROp::div, nat_t(0), param, b);
            auto numerator = world_.op(ROp::mul, nat_t(0), neg_param, a);
            auto denominator = world_.op(ROp::mul, nat_t(0), b, b);
            auto d2 = world_.op(ROp::div, nat_t(0), numerator, denominator);
            B->set_filter(world_.lit_true());
            B->set_body(world_.tuple({d1, d2}));
            return world_.tuple({world_.op(ROp::div, (nat_t)0, a, b), B});
        }
        case ROp::mod: return nullptr;
    }
}

void AutoDiffer::fill_grad(const Def* def, const Def* cur_grad) {
    if (auto lam = def->isa_nominal<Lam>()) {
    }

    if (auto app = def->isa<App>()) {
        if (dst_to_pullback_.contains(app)) {
            auto pb = dst_to_pullback_[app];
            auto grads = world_.app(pb, cur_grad, {"∇" + app->callee()->name()});
            for (size_t i = 0; i < app->num_args(); ++i) {
                fill_grad(app->arg(i), world_.extract(grads, i, {"∇" + std::to_string(i)}));
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
        if (auto param = extract->tuple()->isa<Param>()) {
            add_part_grad(extract, cur_grad);
        } else {
            fill_grad(extract->tuple(), cur_grad);
        }
    }

    if (auto insert = def->isa<Insert>()) {
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
    auto new_part_grad = world_.op(ROp::add, nat_t(0), old_part_grad, part_grad, {"∂" + part_grad->name()});
    dst_to_part_diff_[primal_def] = new_part_grad;

    return new_part_grad;
}

const Def* AutoDiffer::seen(const Def* def) { return src_to_dst_.contains(def) ? src_to_dst_[def] : nullptr; }

////////////////////////////////////////////////////////////////////////////////
// OLD STUFF
////////////////////////////////////////////////////////////////////////////////

/*
class AutoDiffImpl_OLD {
public:
    AutoDiffImpl_OLD(Lam* src_lam, Lam* dst_lam);

    void fill_dst_lam();

private:
    const Def* emit_J_wrapper(const Def* def);
    const Def* emit_axiom_pullback(const Axiom* axiom, const Def* op1, const Def* op2);
    void emit_partial_grad(const Def* def, const Def* res_grad);
    const Def* pack_param_grads(const Def* mem);

    size_t num_params() const { return _src_lam->num_params(); }
    const Def* src_param(size_t i) { return _src_lam->param(i); }
    const Def* dst_param(size_t i) { return _dst_lam->param(i, {src_param(i)->name()}); }
    const Def* isa_dst_param(const Def* def);

    World& _world;
    Lam* _src_lam;
    Lam* _dst_lam;
    Lam* _pb_lam;
    Def2Def _src_to_dst;
    Def2Def _dst_to_pullback;
    Def2Def _dst_to_parts;
};

AutoDiffImpl_OLD::AutoDiffImpl_OLD(Lam* src_lam, Lam* dst_lam)
    : _world(src_lam->world())
    , _src_lam(src_lam)
    , _dst_lam(dst_lam)
    , _pb_lam(_world.lam(
          _dst_lam->type()->as<Pi>()->domain()->ops().back()->as<Pi>()->domain()->ops().back()->as<Pi>(),
          {"φ" + _src_lam->name()})) {
    for (size_t i = 0; i < num_params(); ++i) {
        _src_to_dst[src_param(i)] = dst_param(i);

        if (i > 0 && i < num_params() - 1) {
            _dst_to_parts[dst_param(i)] = _world.lit_real(r64(0), {"∂" + dst_param(i)->name()});
        }
    }
}

const Def* AutoDiffImpl_OLD::isa_dst_param(const Def* def) {
    for (size_t i = 0, e = num_params(); i < e; ++i) {
        if (def == dst_param(i))
            return def;
    }
    return nullptr;
}

const Def* AutoDiffImpl_OLD::pack_param_grads(const Def* mem) {
    Array<const Def*> grads{num_params() - 2, [&](auto i) { return _dst_to_parts[dst_param(i + 1)]; }};
    return _world.tuple({mem, _world.tuple(grads)});
}

void AutoDiffImpl_OLD::fill_dst_lam() {
    _dst_lam->set_filter(_world.lit_false());
    _dst_lam->set_body(emit_J_wrapper(_src_lam->body()));

    THORIN_BREAK;

    emit_partial_grad(_dst_lam->body(), _pb_lam->param(1, {"∇" + _src_lam->name()}));

    auto pb_ret = _pb_lam->ret_param({"return"});
    auto pb_mem = _pb_lam->mem_param({"mem"});
    _pb_lam->set_filter(_world.lit_true());
    _pb_lam->set_body(_world.app(pb_ret, pack_param_grads(pb_mem)));

    scope(_dst_lam);
    scope(_pb_lam);
}

const Def* AutoDiffImpl_OLD::emit_J_wrapper(const Def* def) {
    if (_src_to_dst.contains(def))
        return _src_to_dst[def];

    if (auto tuple = def->isa<Tuple>()) {
        Array<const Def*> defs(tuple->num_ops());
        for (size_t i = 0, e = defs.size(); i < e; ++i) {
            defs[i] = emit_J_wrapper(tuple->op(i));
        }
        return _src_to_dst[def] = _world.tuple(defs);
    }

    if (auto pack = def->isa<Pack>()) {
        return _src_to_dst[def] = _world.pack(pack->arity(), emit_J_wrapper(pack->body()));
    }

    if (auto app = def->isa<App>()) {
        auto callee = app->callee();
        auto arg = emit_J_wrapper(app->arg());

        if (callee == _src_lam->ret_param()) {
            return _world.app(_dst_lam->ret_param(), arg);
        }

        if (callee->type()->as<Pi>()->is_cn()) {
            auto rd_callee = _world.op_rev_diff(callee);

            auto cn = _world.lam(rd_callee->type()->as<Pi>()->domains().back()->as<Pi>(), {"rd_cn"});
            Array<const Def*> args(arg->ops());
            args.back() = cn;

            auto [mem, res, B] = cn->param()->split<3>();
            _dst_to_pullback[res] = B;

            cn->set_filter(_world.lit_true());
            cn->set_body(_world.app(arg->ops().back(), {mem, res}));

            return _src_to_dst[def] = _world.app(rd_callee, args);
        } else {
            if (auto axiom_app = callee->isa<App>()) {
                if (auto axiom = axiom_app->callee()->isa<Axiom>()) {
                    if (axiom->tag() == flags_t(Tag::ROp)) {
                        auto [op1, op2] = arg->split<2>();
                        auto [res, B] = emit_axiom_pullback(axiom, op1, op2)->split<2>();

                        _dst_to_pullback[res] = B;
                        return _src_to_dst[def] = res;
                    }
                }
            }
        }
    }

    if (auto lam = def->isa<Lam>()) {
        if (lam->type()->as<Pi>()->is_cn()) {
            auto rd_lam = _world.lam(lam->type()->as<Pi>(), {"rd_" + lam->name()});
            rd_lam->set_filter(lam->filter());
            rd_lam->set_body(emit_J_wrapper(lam->body()));
            return _src_to_dst[def] = rd_lam;
        }
    }

    errf("Not differentiable: {}", def);
    return def;
}

const Def* AutoDiffImpl_OLD::emit_axiom_pullback(const Axiom* axiom, const Def* op1, const Def* op2) {
    assert(op1->type() == op2->type());

    auto r_type = op1->type();
    auto pi = _world.pi(r_type, _world.sigma({r_type, r_type}));

    switch (ROp(axiom->flags())) {
        // ∇(a + b) = λ∂f.[∂f, ∂f]
        case ROp::add: {
            auto B = _world.lam(pi, {"φ+"});
            auto param = B->param();
            B->set_filter(_world.lit_true());
            B->set_body(_world.tuple({param, param}));
            return _world.tuple({_world.op(ROp::add, (nat_t)0, op1, op2), B});
        }
        // ∇(a - b) = λ∂f.[∂f, -∂f]
        case ROp::sub: {
            auto B = _world.lam(pi, {"φ-"});
            auto param = B->param();
            B->set_filter(_world.lit_true());
            B->set_body(_world.tuple({param, _world.op_ROp_minus((nat_t)0, param)}));
            return _world.tuple({_world.op(ROp::sub, (nat_t)0, op1, op2), B});
        }
        // ∇(a * b) = λ∂f.[∂f*b, ∂f*a]
        case ROp::mul: {
            auto B = _world.lam(pi, {"φ*"});
            auto param = B->param();
            auto d1 = _world.op(ROp::mul, nat_t(0), param, op2);
            auto d2 = _world.op(ROp::mul, nat_t(0), param, op1);
            B->set_filter(_world.lit_true());
            B->set_body(_world.tuple({d1, d2}));
            return _world.tuple({_world.op(ROp::mul, (nat_t)0, op1, op2), B});
        }
        // ∇(a / b) = λ∂f.[∂f/b, (-∂f*a)/(b²)]
        case ROp::div: {
            auto B = _world.lam(pi, {"φ/"});
            auto param = B->param();
            auto neg_param = _world.op_ROp_minus(nat_t(0), B->param());
            auto d1 = _world.op(ROp::div, nat_t(0), param, op2);
            auto numerator = _world.op(ROp::mul, nat_t(0), neg_param, op1);
            auto denominator = _world.op(ROp::mul, nat_t(0), op2, op2);
            auto d2 = _world.op(ROp::div, nat_t(0), numerator, denominator);
            B->set_filter(_world.lit_true());
            B->set_body(_world.tuple({d1, d2}));
            return _world.tuple({_world.op(ROp::div, (nat_t)0, op1, op2), B});
        }
        case ROp::mod: return nullptr;
    }
}

void AutoDiffImpl_OLD::emit_partial_grad(const Def* def, const Def* res_grad) {
    if (auto param = isa_dst_param(def)) {
        _dst_to_parts[param] =
            _world.op(ROp::add, nat_t(0), res_grad, _dst_to_parts[param], {"∂" + param->name()});
    }

    if (auto tuple = def->isa<Tuple>()) {
        Array<const Def*> defs(tuple->num_ops());
        for (size_t i = 0, e = defs.size(); i < e; ++i) {
            emit_partial_grad(tuple->op(i), res_grad);
        }
    }

    if (auto pack = def->isa<Pack>()) {
        emit_partial_grad(pack->body(), res_grad);
    }

    if (auto app = def->isa<App>()) {
        if (auto axiom_app = app->callee()->isa<App>()) {
            if (auto axiom = axiom_app->callee()->isa<Axiom>()) {
                if (axiom->tag() == flags_t(Tag::ROp)) {
                    auto B = _dst_to_pullback[app];
                    auto grads = _world.app(B, res_grad, {"∇"});

                    for (size_t i = 0, e = app->num_args(); i < e; ++i) {
                        emit_partial_grad(app->arg(i), _world.extract(grads, i, {"∇" + std::to_string(i)}));
                    }
                }
            }
        }
    }
}*/

} // namespace

const Def* AutoDiff::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto type_app = app->callee()->isa<App>()) {
            if (auto axiom = type_app->callee()->isa<Axiom>(); axiom && axiom->tag() == Tag::RevDiff) {
                auto src_lam = app->arg(0)->as_nominal<Lam>();
                auto src_pi = src_lam->type()->as<Pi>();
                auto& world = src_lam->world();

                auto dst_pi = app->type()->as<Pi>();
                auto dst_lam = world.lam(dst_pi, {"rev_diff_" + src_lam->name()});

                auto pb_pi = dst_pi->domains().back()->as<Pi>();
                auto pb_lam = world.lam(pb_pi, {"pullback_" + src_lam->name()});
                auto [pb_mem, pb_grad, pb_ret] = pb_lam->param()->split<3>();

                auto ret_pi = src_pi->domains().back()->as<Pi>();
                auto ret_lam = world.lam(ret_pi, {"wrapper"});

                Def2Def src_to_dst;
                for (size_t i = 0, e = src_lam->num_params(); i < e; ++i) {
                    auto src_param = src_lam->param(i);
                    auto dst_param = dst_lam->param(i, {src_param->name()});
                    src_to_dst[src_param] = i == e - 1 ? ret_lam : dst_param;
                }
                auto differ = AutoDiffer{world, pb_grad, src_to_dst};

                dst_lam->set_filter(src_lam->filter());
                dst_lam->set_body(differ.reverse_diff(src_lam->body()));

                auto num_ret_body_args = dst_pi->domains().back()->as<Pi>()->num_domains();
                Array<const Def*> ret_body_args{num_ret_body_args,
                    [&](auto i) { return i < num_ret_body_args - 1 ? ret_lam->param(i) : pb_lam; }};
                ret_lam->set_filter(world.lit_true());
                ret_lam->set_body(world.app(dst_lam->ret_param({"return"}), ret_body_args));

                auto num_grads = src_lam->num_params() - 2;
                Array<const Def*> grads{num_grads, [&](auto i) { return differ.grad(dst_lam->param(i + 1)); }};
                pb_lam->set_filter(world.lit_true());
                pb_lam->set_body(world.app(pb_lam->ret_param(), {pb_lam->mem_param(), world.tuple(grads)}));

                return dst_lam;
            }
        }
    }

    return def;
}

} // namespace thorin
