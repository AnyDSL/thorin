#include "auto_diff.h"

#include "../analyses/scope.h"

namespace thorin {

namespace {
class AutoDiffImpl {
public:
    AutoDiffImpl(Lam* src_lam, Lam* dst_lam);

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

AutoDiffImpl::AutoDiffImpl(Lam* src_lam, Lam* dst_lam)
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

const Def* AutoDiffImpl::isa_dst_param(const Def* def) {
    for (size_t i = 0, e = num_params(); i < e; ++i) {
        if (def == dst_param(i))
            return def;
    }
    return nullptr;
}

const Def* AutoDiffImpl::pack_param_grads(const Def* mem) {
    Array<const Def*> grads{num_params() - 2, [&](auto i) { return _dst_to_parts[dst_param(i + 1)]; }};
    return _world.tuple({mem, _world.tuple(grads)});
}

void AutoDiffImpl::fill_dst_lam() {
    auto res = emit_J_wrapper(_src_lam->body()->as<App>()->arg()->op(1)); // op(1) to skip the mem…
    emit_partial_grad(res, _pb_lam->param(1));

    auto dst_ret = _dst_lam->ret_param({"return"});
    auto dst_mem = _dst_lam->mem_param({"mem"});
    _dst_lam->set_filter(_world.lit_false());
    _dst_lam->set_body(_world.app(dst_ret, {dst_mem, res, _pb_lam}));

    auto pb_ret = _pb_lam->ret_param({"return"});
    auto pb_mem = _pb_lam->mem_param({"mem"});
    _pb_lam->set_filter(_world.lit_true());
    _pb_lam->set_body(_world.app(pb_ret, pack_param_grads(pb_mem)));
}

const Def* AutoDiffImpl::emit_J_wrapper(const Def* def) {
    if (_src_to_dst.contains(def))
        return _src_to_dst[def];

    if (auto tuple = def->isa<Tuple>()) {
        Array<const Def*> defs(tuple->num_ops());
        for (size_t i = 0, e = defs.size(); i < e; ++i) {
            defs[i] = emit_J_wrapper(tuple->op(i));
        }
        return _src_to_dst[def] = _world.tuple(defs);
    }

    if (auto app = def->isa<App>()) {
        auto arg = emit_J_wrapper(app->arg());

        if (auto axiom_app = app->callee()->isa<App>()) {
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

    THORIN_UNREACHABLE;
}

const Def* AutoDiffImpl::emit_axiom_pullback(const Axiom* axiom, const Def* op1, const Def* op2) {
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
            auto B = _world.lam(pi, {"φ*"});
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

void AutoDiffImpl::emit_partial_grad(const Def* def, const Def* res_grad) {
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
}

} // namespace

const Def* AutoDiff::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto type_app = app->callee()->isa<App>()) {
            if (auto axiom = type_app->callee()->isa<Axiom>(); axiom && axiom->tag() == Tag::RevDiff) {
                auto src_lam = app->arg(0)->as_nominal<Lam>();
                auto dst_lam = src_lam->world().lam(app->type()->as<Pi>(), {"rev_diff_" + src_lam->name()});

                AutoDiffImpl(src_lam, dst_lam).fill_dst_lam();

                return dst_lam;
            }
        }
    }

    return def;
}

} // namespace thorin
