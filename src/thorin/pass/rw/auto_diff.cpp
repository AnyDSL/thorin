#include "thorin/pass/rw/auto_diff.h"

#include <algorithm>
#include <string>

#include "thorin/analyses/scope.h"

namespace thorin {

#define THORIN_CONV(m) m(Conv, s2s) m(Conv, u2u) m(Conv, s2r) m(Conv, u2r) m(Conv, r2s) m(Conv, r2u) m(Conv, r2r)
// Sadly, we need to "unpack" the type
const Def* lit_of_type(World& world, const Def* type, u64 lit) {
    // TODO: Actually implement this. For now, all functions are r32 anyways, so whatever.

    if (auto real = isa<Tag::Real>(type))
        return world.lit_real(as_lit(real->arg()), lit);
    return world.lit_int(as_lit(as<Tag::Int>(type)), lit);
}

const Def* ONE(World& world, const Def* def) { return lit_of_type(world, def, 1); }
const Def* ZERO(World& world, const Def* def) { return lit_of_type(world, def, 0); }

namespace {

class AutoDiffer {
public:
    AutoDiffer(World& world, Lam* initialpb, const Def2Def src_to_dst, const Def* A)
        : world_{world}
        , initialpb_{initialpb}
        , src_to_dst_{src_to_dst}
        , idpb{}
    {
        auto idpi = world_.pi(A, A);
        idpb = world_.nom_lam(idpi, world_.dbg("id"));
        idpb->set_filter(world_.lit_true());
        idpb->set_body(idpb->var());
    }

    const Def* reverse_diff(Lam* src);
    const Def* forward_diff(const Def*) { throw "not implemented"; }

private:
    const Def* j_wrap(const Def* def);
    const Def* j_wrap_rop(ROp op, const Def* a, const Def* b);

    const Def* seen(const Def* src);

    World& world_;
    Lam* initialpb_;
    Def2Def src_to_dst_;
    Lam* idpb;
    Def2Def pullbacks_;  // <- maps a *copied* src term to its pullback function
};

const Def* AutoDiffer::reverse_diff(Lam* src) {
    // For each param, create an appropriate pullback. It is just the identity function for each of those.
    for(size_t i = 0, e = src->num_vars(); i < e; ++i) {
        auto src_param = src->var(i);
        if(src_param == src->ret_var() || src_param == src->mem_var()) {
            continue;
        }
        pullbacks_[src_to_dst_[src_param]] = idpb;
    }
    auto dst = j_wrap(src->body());

    // TODO: this assumes functions ℝ →  ℝ
    auto pb_invoke = world_.app(pullbacks_[dst], initialpb_->var(1), world_.dbg("pb_invoke"));
    initialpb_->set_body(world_.app(initialpb_->ret_var(), {initialpb_->mem_var(), pb_invoke}));

    return dst;
}

// We implement AD in a similar way as described by Brunel et al., 2020
//  <x², λa. x'(a * 2*x)>
//       ^^^^^^^^^- pullback. The intuition is as follows:
//                            Each value x has a pullback pb_x.
//                            pb_x receives a value that was differentiated with respect to x.
//                  Thus, the "initial" pullback for parameters must be the identity function.
// Here is a very brief example of what should happen in `j_wrap` and `j_wrap_rop`:
//  
//      SOURCE             |  PRIMAL VERSION OF SOURCE
//   ----------------------+-----------------------------------------------------------------------
//     // x is parameter   | // <x,x'> is parameter. x' should be something like λz.z
//    let y = 3 * x * x;   | let <y,y'> = <3 * x * x, λz. x'(z * (6 * x))>;
//    y * x                | <y * x, λz. y'(z * x) + x'(z * y)>
//
// Instead of explicitly putting everything into a pair, we just use the pullbacks freely
//  Each `x` gets transformed to a `<x, λδz. δz * (δz / δx)>`
const Def* AutoDiffer::j_wrap(const Def* def) {
    if (auto dst = seen(def))
        return dst;

    if (auto var = def->isa<Var>()) {
        errf("Out of scope var: {}\n Not differentiable", var);
        THORIN_UNREACHABLE;
    }
    if (auto axiom = def->isa<Axiom>()) {
        errf("Axioms are not differentiable. Found axiom: {}", axiom);
        THORIN_UNREACHABLE;
    }
    if (auto app = def->isa<App>()) {
        auto callee = app->callee();
        auto arg = app->arg();

        // Handle binary operations
        if (auto inner = callee->isa<App>()) {
            // Take care of binary operations
            if (auto axiom = inner->callee()->isa<Axiom>()) {
                if (axiom->tag() == Tag::ROp) {
                    auto [a, b] = j_wrap(arg)->split<2>();
                    auto dst = j_wrap_rop(ROp(axiom->flags()), a, b);
                    src_to_dst_[app] = dst;
                    return dst;
                }

                if (axiom->tag() == Tag::RCmp) {
                    auto [a, b] = j_wrap(arg)->split<2>();
                    auto dst = world_.op(RCmp(axiom->flags()), nat_t(0), a, b);
                    src_to_dst_[app] = dst;
                    return world_.tuple({inner, dst});
                }
            }
        }
        // Handle function calls
        if (callee->type()->as<Pi>()->is_returning()) {
            // we simply recursively apply the whole autodiff pass to the thing we want to call
            // Thus, a `cn [:mem, A, cn[:mem, B]]` becomes `cn [:mem, A, B, cn[:mem, <B, A>]`
            // So, since call site has changed, we need a wrapper that takes care of calling
            auto rd_callee = world_.op_rev_diff(callee);
            auto rd_pi = rd_callee->type()->as<Pi>();

            // We want to call rd_callee and store the result in dst and grab its pullback for pullbacks_
            // Unfortunately, the call-site abstracted the pullback away, it's squished together in the
            // function's signature, i.e. instead of `fn(f32) -> <f32, fn(f32) -> f32>` we have `fn(f32, f32) -> <f32, f32>`
            // Thus, we need to build a wrapper that represents the pullback
            // To this end, we leaverage the chain rule once again.

            // First things first, call the function.
            auto ad = j_wrap(arg);

            // TODO: this assumes ℝ →  ℝ
            auto get_result_pi = world_.cn(world_.sigma({world_.type_mem(), world_.tuple({rd_pi->dom(1), rd_pi->dom(2)})}));
            auto get_result = world_.nom_lam(get_result_pi, world_.dbg("get_result"));
            auto call = world_.app(rd_callee, {arg, ad, get_result});

            auto y = get_result->var(1, world_.dbg("y"));
            auto dy = get_result->var(2, world_.dbg("δy"));
            // now that we have the result, we  ....
            assert(false);
        }
        auto cd = j_wrap(callee);
        auto ad = j_wrap(arg);
        auto ad_mem = world_.extract(ad, (u64)0, world_.dbg("mem"));
        auto ad_arg = world_.extract(ad, (u64)1, world_.dbg("arg"));
        auto dst = world_.app(cd, {ad_mem, ad_arg, pullbacks_[ad]});
        src_to_dst_[app] = dst;
        pullbacks_[dst] = pullbacks_[ad];   // <-- FIXME: probably not correct in general
        return dst;
    }

    if (auto tuple = def->isa<Tuple>()) {
        Array<const Def*> ops{tuple->num_ops(), [&](auto i) { return j_wrap(tuple->op(i)); }};
        auto dst = world_.tuple(ops);
        src_to_dst_[tuple] = dst;

        // special care to wire up the pullbacks. TODO: this assumes functions ℝ →  ℝ
        if(ops.size() == 2 && ops[0]->type() == world_.type_mem()) {
            pullbacks_[dst] = pullbacks_[ops[1]];
        }
        else {
            // fallback
            pullbacks_[dst] = idpb;
        }
        return dst;
    }

    if (auto pack = def->isa<Pack>()) {
        auto dst = world_.pack(pack->type()->arity(), j_wrap(pack->body()));
        src_to_dst_[pack] = dst;
        pullbacks_[dst] = idpb;
        return dst;
    }

    if (auto extract = def->isa<Extract>()) {
        auto dst = world_.extract(j_wrap(extract->tuple()), j_wrap(extract->index()));
        src_to_dst_[extract] = dst;
        pullbacks_[dst] = idpb;
        return dst;
    }

    if (auto insert = def->isa<Insert>()) {
        auto dst = world_.insert(j_wrap(insert->tuple()), j_wrap(insert->index()), j_wrap(insert->value()));
        src_to_dst_[insert] = dst;
        pullbacks_[dst] = idpb;
        return dst;
    }

    if (auto lit = def->isa<Lit>()) {
        // The derivative of a literal is ZERO
        auto zeropi = world_.pi(lit->type(), lit->type());
        auto zeropb = world_.nom_lam(zeropi, world_.dbg("id"));
        zeropb->set_filter(world_.lit_true());
        zeropb->set_body(ZERO(world_, lit->type()));
        pullbacks_[lit] = zeropb;
        return lit;
    }

    errf("Not handling: {}", def);
    THORIN_UNREACHABLE;
}

const Def* AutoDiffer::j_wrap_rop(ROp op, const Def* a, const Def* b) {
    // build up pullback type for this expression
    auto r_type = a->type();
    auto pbpi = world_.pi(r_type, r_type);
    auto pb = world_.nom_lam(pbpi, world_.dbg("φ"));
    pb->set_filter(world_.lit_true());

    auto one = ONE(world_, r_type);

    // Grab argument pullbacks
    auto apb = pullbacks_[a];
    auto bpb = pullbacks_[b];
    switch (op) {
        // ∇(a + b) = λz.∂a(z * (1 + 0)) + ∂b(z * (0 + 1))
        case ROp::add: {
            auto dst = world_.op(ROp::add, (nat_t)0, a, b);
            auto var = pb->var();
            pb->set_dbg(world_.dbg(pb->name() + "+"));

            auto adiff = world_.app(apb, world_.op(ROp::mul, (nat_t)0, var, one));
            auto bdiff = world_.app(bpb, world_.op(ROp::mul, (nat_t)0, var, one));

            pb->set_body(world_.op(ROp::add, (nat_t)0, adiff, bdiff));
            pullbacks_[dst] = pb;

            return dst;
        }
        // ∇(a - b) = λz.∂a(z * (0 + 1)) - ∂b(z * (0 + 1))
        case ROp::sub: {
            auto dst = world_.op(ROp::sub, (nat_t)0, a, b);
            auto var = pb->var();
            pb->set_dbg(world_.dbg(pb->name() + "-"));

            auto adiff = world_.app(apb, world_.op(ROp::mul, (nat_t)0, var, one));
            auto bdiff = world_.app(bpb, world_.op(ROp::mul, (nat_t)0, var, world_.op_rminus((nat_t)0, one)));

            pb->set_body(world_.op(ROp::add, (nat_t)0, adiff, bdiff));
            pullbacks_[dst] = pb;

            return dst;
        }
        // ∇(a * b) = λz.∂a(z * (1 * b + a * 0)) + ∂b(z * (0 * b + a * 1))
        //          potential opt: if ∂a = ∂b, do: ∂a(z * (a + b))
        //             do this in the future. We need to make sure the pb is linear.
        //             This should be doable without additional tracking if we change
        //             their types from `R -> R` to `R -> ⊥`
        case ROp::mul: {
            auto dst = world_.op(ROp::mul, (nat_t)0, a, b);
            auto var = pb->var();
            pb->set_dbg(world_.dbg(pb->name() + "*"));

            auto adiff = world_.app(apb, world_.op(ROp::mul, (nat_t)0, var, b));
            auto bdiff = world_.app(bpb, world_.op(ROp::mul, (nat_t)0, var, a));

            pb->set_body(world_.op(ROp::add, (nat_t)0, adiff, bdiff));
            pullbacks_[dst] = pb;

            return dst;
        }
        // ∇(a / b) = λz.∂a ∂b
        case ROp::div: {
            THORIN_UNREACHABLE; // TODO
        }
        default:
            THORIN_UNREACHABLE;
    }
}

const Def* AutoDiffer::seen(const Def* def) { return src_to_dst_.contains(def) ? src_to_dst_[def] : nullptr; }

} // namespace

const Def* AutoDiff::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto type_app = app->callee()->isa<App>()) {
            if (auto axiom = type_app->callee()->isa<Axiom>(); axiom && axiom->tag() == Tag::RevDiff) {
                auto src_lam = app->arg(0)->as_nom<Lam>();
                // this should be something like `cn[:mem, r32, cn[:mem, r32]]`
                auto& world = src_lam->world();

                // Copy of the original function, augmented to return <r32, f(32)->r32>.
                // So, for a src_lam of type `cn[:mem, r32, cn[:mem, r32]]` 
                // we have a base_dst_pi of type `cn[:mem, r32, r32, cn[:mem, <<2:nat, r32>>]]`
                auto base_dst_pi = app->type()->as<Pi>();
                auto base_dst_lam = world.nom_lam(base_dst_pi, world.dbg("top_level_rev_diff_" + src_lam->name()));

                // build up a copy of src_pi that wraps around base_dst_pi.
                // We want for `A -> B` the type `A -> (B * (B -> A))`.
                //  i.e. cn[:mem, A, [:mem, B]] ---> cn[:mem, A, cn[:mem, B, cn[:mem, B, A]]]
                auto A = base_dst_pi->dom(1);
                auto B = base_dst_pi->dom(2); // FIXME: this assumes funs from ℝ to ℝ
                auto dst_pi = world.cn_mem_flat(A, world.sigma({B, world.pi(B, A)})); 
                auto dst_lam = world.nom_lam(dst_pi, world.dbg("rev_diff_" + src_lam->name()));

                // We now take care of this:
                //   sq(x, δy) -> (y, δx) {    <- this is "base_dst_lam"
                //     (y, pb) <- sq_cpy(x)    <- sq_cpy is "dst_lam"
                //     δx <- pb(δy)
                //     (y, δx)
                //   }
                //
                base_dst_lam->set_filter(world.lit_true());
                // sq(x, δy)
                auto x = base_dst_lam->var(1, world.dbg("x")); // A
                auto del_y = base_dst_lam->var(2, world.dbg("δy")); // B // FIXME: this assumes that there is only one seed value

                // The actual AD, i.e. construct "sq_cpy"
                // This is a wrapper around the chain of pullbacks
                auto derivative_map_pi = world.cn_mem_flat(B, A);
                auto derivative_map_lam = world.nom_lam(derivative_map_pi, world.dbg("derivative_map"));
                derivative_map_lam->set_filter(world.lit_true());

                // Perform the actual AD now
                Def2Def src_to_dst;
                for (size_t i = 0, e = src_lam->num_vars(); i < e; ++i) {
                    auto src_param = src_lam->var(i);
                    auto dst_param = dst_lam->var(i, world.dbg(src_param->name()));
                    src_to_dst[src_param] = i == e - 1 ? dst_lam->ret_var() : dst_param;
                }
                auto differ = AutoDiffer{world, derivative_map_lam, src_to_dst, A};
                dst_lam->set_filter(src_lam->filter());
                dst_lam->set_body(differ.reverse_diff(src_lam));

                // (y, pb) <- sq_cpy(x)  a bit more complicated due to cps
                auto fnresult_and_pullback_cnpi = dst_pi->dom()->op(dst_pi->dom()->num_ops() - 1)->as<Pi>();
                auto fnresult_and_pullback_cn = world.nom_lam(fnresult_and_pullback_cnpi, world.dbg("ypb_cn"));
                fnresult_and_pullback_cn->set_filter(world.lit_true());
                
                base_dst_lam->set_body(world.app(dst_lam, {base_dst_lam->mem_var(), x, fnresult_and_pullback_cn}));
                auto y = fnresult_and_pullback_cn->var(1, world.dbg("y"));
                auto pb = fnresult_and_pullback_cn->var(2, world.dbg("pb"));

                // δx <- pb(δy)
                auto del_x = world.app(pb, del_y);

                // return (y, δx)
                // the top-level expects the tuple (y, δx), we just use y here as free var, pulling it in our scope
                auto tup = world.tuple({y, del_x});
                fnresult_and_pullback_cn->set_body(world.app(base_dst_lam->ret_var(world.dbg("top_level_return")), {fnresult_and_pullback_cn->mem_var(), tup}));


                debug_dump(base_dst_lam);
                debug_dump(derivative_map_lam);

                return base_dst_lam;
            }
        }
    }

    return def;
}

}
