#include "thorin/pass/rw/auto_diff.h"

#include <algorithm>
#include <string>

#include "thorin/analyses/scope.h"

namespace thorin {

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
    AutoDiffer(World& world, const Def2Def src_to_dst, const Def* A)
        : world_{world}
        , src_to_dst_{src_to_dst}
        , idpb{}
    {
        auto idpi = world_.cn_mem_flat(A, A);
        idpb = world_.nom_lam(idpi, world_.dbg("id"));
        idpb->set_filter(world_.lit_true());
        idpb->set_body(world_.app(idpb->ret_var(), {idpb->mem_var(), idpb->var(1, world.dbg("a"))}));
    }

    const Def* reverse_diff(Lam* src);
    const Def* forward_diff(const Def*) { throw "not implemented"; }
private:
    const Def* j_wrap(const Def* def);
    const Def* j_wrap_rop(ROp op, const Def* a, const Def* b);

    const Def* seen(const Def* src);

    // chains cn[:mem, A, cn[:mem, B]] and cn[:mem, B, cn[:mem, C]] to a toplevel cn[:mem, A, cn[:mem, C]]
    Lam* chain(Lam* a, Lam* b);

    World& world_;
    Def2Def src_to_dst_;
    Lam* idpb;
    DefMap<const Def*> pullbacks_;  // <- maps a *copied* src term to its pullback function
};

Lam* AutoDiffer::chain(Lam* a, Lam* b) {
    // chaining with identity is neutral
    if (a == idpb) return b;
    if (b == idpb) return a;

    auto at = a->type()->as<Pi>();
    auto bt = b->type()->as<Pi>();

    auto A = at->doms()[1];
    auto B = bt->doms()[1];
    auto C = bt->doms()[2]->as<Pi>()->doms()[1];

    auto pi = world_.cn_mem_flat(A, C);
    auto toplevel = world_.nom_lam(pi, world_.dbg("chain"));

    auto middlepi = world_.cn({world_.type_mem(), B});
    auto middle = world_.nom_lam(middlepi, world_.dbg("chain"));

    toplevel->set_body(world_.app(a, {toplevel->mem_var(), toplevel->var(1), middle}));
    middle->set_body(world_.app(b, {middle->mem_var(), middle->var(1), toplevel->ret_var()}));

    toplevel->set_filter(world_.lit_true());
    middle->set_filter(world_.lit_true());

    return toplevel;
}


const Def* AutoDiffer::reverse_diff(Lam* src) {
    // For each param, create an appropriate pullback. It is just the identity function for each of those.
    for(size_t i = 0, e = src->num_vars(); i < e; ++i) {
        auto src_param = src->var(i);
        if(src_param == src->ret_var() || src_param == src->mem_var()) {
            continue;
        }
        auto dst = src_to_dst_[src_param];
        pullbacks_[dst] = idpb;
    }
    auto dst = j_wrap(src->body());
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
    if (auto lam = def->isa_nom<Lam>()) {
        // FIXME: pb type correct? might not be able to just use idpb->type() here
        auto old_pi = lam->type()->as<Pi>();
        auto pi = world_.cn({world_.type_mem(), old_pi->doms()[1], idpb->type()});
        auto dst = world_.nom_lam(pi, world_.dbg(lam->name()));
        src_to_dst_[lam->var()] = dst->var();
        pullbacks_[dst->var()] = dst->var(dst->num_vars() - 1);
        dst->set_filter(lam->filter());

        auto bdy = j_wrap(lam->body());
        dst->set_body(bdy);
        src_to_dst_[lam] = dst;
        pullbacks_[dst] = pullbacks_[bdy];
        return dst;
    }
    if (auto app = def->isa<App>()) {
        auto callee = app->callee();
        auto arg = app->arg();

        // Handle binary operations
        if (auto inner = callee->isa<App>()) {
            // Take care of binary operations
            if (auto axiom = inner->callee()->isa<Axiom>()) {
                if (axiom->tag() == Tag::ROp) {
                    auto [a, b] = j_wrap(arg)->outs<2>();
                    auto dst = j_wrap_rop(ROp(axiom->flags()), a, b);
                    src_to_dst_[app] = dst;
                    return dst;
                }

                if (axiom->tag() == Tag::RCmp) {
                    auto [a, b] = j_wrap(arg)->outs<2>();
                    auto dst = world_.op(RCmp(axiom->flags()), nat_t(0), a, b);
                    src_to_dst_[app] = dst;
                    return world_.tuple({inner, dst});
                }
            }
        }
        auto ad = j_wrap(arg);
        auto ad_mem = world_.extract(ad, (u64)0, world_.dbg("mem"));
        auto ad_arg = world_.extract(ad, (u64)1, world_.dbg("arg"));

        auto cpi = (src_to_dst_.count(callee) ? src_to_dst_[callee]->type()->as<Pi>() : nullptr);
        if(cpi != nullptr) {
            // check if our functions returns a pullback already
            if (auto rett = cpi->doms().back()->isa<Pi>(); rett && rett->is_returning()) {
                auto cd = j_wrap(callee);

                if (pullbacks_.count(ad)) {
                    auto dst = world_.app(cd, {ad_mem, ad_arg, pullbacks_[ad]});
                    src_to_dst_[app] = dst;

                    pullbacks_[dst] = pullbacks_[ad];
                    return dst;
                }
                else {
                    assert(ad->num_outs() == arg->num_outs() + 1 && "Pullback must have been added here.");

                    auto dst = world_.app(cd, ad);
                    src_to_dst_[app] = dst;

                    return dst;
                }
            }
        }
        if (!callee->isa_nom<Lam>() && src_to_dst_.count(callee)) {
            auto dstcallee = src_to_dst_[callee];

            auto dst = world_.app(dstcallee, {ad_mem, ad_arg, pullbacks_[ad]});
            pullbacks_[dst] = pullbacks_[ad]; // <- chain pullback of dstcallee?

            return dst;
        }
        auto dst_callee = world_.op_rev_diff(callee);
        auto dst = world_.app(dst_callee, ad);
        pullbacks_[dst] = pullbacks_[ad];

        return dst;
    }

    if (auto tuple = def->isa<Tuple>()) {
        DefArray ops{tuple->num_ops(), [&](auto i) { return j_wrap(tuple->op(i)); }};
        auto dst = world_.tuple(ops);
        src_to_dst_[tuple] = dst;

        // FIXME: this obviously doesn't work in general
        if(ops.size() == 2) {
            pullbacks_[dst] = pullbacks_[ops[1]];
        }
        else {
            // fallback
            pullbacks_[dst] = idpb;
            for (auto i : ops) {
                if (pullbacks_.contains(i))
                    pullbacks_[dst] = pullbacks_[i];
            }
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
        auto jtup = j_wrap(extract->tuple());
        auto dst = world_.extract_unsafe(jtup, extract->index());
        src_to_dst_[extract] = dst;
        pullbacks_[dst] = pullbacks_[jtup]; // <- FIXME: This must not be idpb lmao
        return dst;
    }

    if (auto insert = def->isa<Insert>()) {
        auto dst = world_.insert(j_wrap(insert->tuple()), insert->index(), j_wrap(insert->value()));
        src_to_dst_[insert] = dst;
        pullbacks_[dst] = idpb;
        return dst;
    }

    if (auto lit = def->isa<Lit>()) {
        // The derivative of a literal is ZERO
        auto zeropi = world_.cn_mem_flat(lit->type(), lit->type());
        auto zeropb = world_.nom_lam(zeropi, world_.dbg("id"));
        zeropb->set_filter(world_.lit_true());
        zeropb->set_body(world_.app(zeropb->ret_var(), {zeropb->mem_var(), ZERO(world_, lit->type())}));
        pullbacks_[lit] = zeropb;
        return lit;
    }

    errf("Not handling: {}", def);
    THORIN_UNREACHABLE;
}

const Def* AutoDiffer::j_wrap_rop(ROp op, const Def* a, const Def* b) {
    // build up pullback type for this expression
    auto r_type = a->type();
    auto pbpi = world_.cn_mem_flat(r_type, r_type);
    auto pb = world_.nom_lam(pbpi, world_.dbg("φ"));

    auto middle = world_.nom_lam(world_.cn({world_.type_mem(), r_type}), world_.dbg("φmiddle"));
    auto end = world_.nom_lam(world_.cn({world_.type_mem(), r_type}), world_.dbg("φend"));

    pb->set_filter(world_.lit_true());
    middle->set_filter(world_.lit_true());
    end->set_filter(world_.lit_true());

    auto one = ONE(world_, r_type);

    // Grab argument pullbacks
    auto apb = pullbacks_[a];
    auto bpb = pullbacks_[b];
    switch (op) {
        // ∇(a + b) = λz.∂a(z * (1 + 0)) + ∂b(z * (0 + 1))
        case ROp::add: {
            auto dst = world_.op(ROp::add, (nat_t)0, a, b);
            pb->set_dbg(world_.dbg(pb->name() + "+"));

            pb->set_body(world_.app(apb, {pb->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), one), middle}));
            middle->set_body(world_.app(bpb, {middle->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), one), end}));
            auto adiff = middle->var(1);
            auto bdiff = end->var(1);

            end->set_body(world_.app(pb->ret_var(), {end->mem_var(), world_.op(ROp::add, (nat_t)0, adiff, bdiff)}));
            pullbacks_[dst] = pb;

            return dst;
        }
        // ∇(a - b) = λz.∂a(z * (0 + 1)) - ∂b(z * (0 + 1))
        case ROp::sub: {
            auto dst = world_.op(ROp::sub, (nat_t)0, a, b);
            pb->set_dbg(world_.dbg(pb->name() + "-"));

            pb->set_body(world_.app(apb, {pb->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), one), middle}));
            middle->set_body(world_.app(bpb, {middle->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), world_.op_rminus((nat_t)0, one)), end}));
            auto adiff = middle->var(1);
            auto bdiff = end->var(1);

            end->set_body(world_.app(pb->ret_var(), {end->mem_var(), world_.op(ROp::add, (nat_t)0, adiff, bdiff)}));
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
            pb->set_dbg(world_.dbg(pb->name() + "*"));

            pb->set_body(world_.app(apb, {pb->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), b), middle}));
            middle->set_body(world_.app(bpb, {middle->mem_var(), world_.op(ROp::mul, (nat_t)0, pb->var(1), a), end}));
            auto adiff = middle->var(1);
            auto bdiff = end->var(1);

            end->set_body(world_.app(pb->ret_var(), {end->mem_var(), world_.op(ROp::add, (nat_t)0, adiff, bdiff)}));
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

                // We get for `A -> B` the type `A -> (B * (B -> A))`.
                //  i.e. cn[:mem, A, [:mem, B]] ---> cn[:mem, A, cn[:mem, B, cn[:mem, B, A]]]
                auto dst_pi = app->type()->as<Pi>();
                auto dst_lam = world.nom_lam(dst_pi, world.dbg("top_level_rev_diff_" + src_lam->name()));
                dst_lam->set_filter(src_lam->filter());
                auto A = dst_pi->dom(1);
                auto B = src_lam->ret_var()->type()->as<Pi>()->dom(1);

                // The actual AD, i.e. construct "sq_cpy"
                Def2Def src_to_dst;
                for (size_t i = 0, e = src_lam->num_vars(); i < e; ++i) {
                    auto src_param = src_lam->var(i);
                    auto dst_param = dst_lam->var(i, world.dbg(src_param->name()));
                    src_to_dst[src_param] = i == e - 1 ? dst_lam->ret_var() : dst_param;
                }
                auto differ = AutoDiffer{world, src_to_dst, A};
                dst_lam->set_body(world.lit_true());
                dst_lam->set_body(differ.reverse_diff(src_lam));

                //debug_dump(dst_lam);

                return dst_lam;
            }
        }
    }

    return def;
}

}
