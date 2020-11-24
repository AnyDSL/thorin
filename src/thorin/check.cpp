#include "thorin/check.h"

#include "thorin/world.h"

namespace thorin {

bool Checker::equiv(const Def* d1, const Def* d2) {
    if (d1 == d2 || (!d1->is_set() && !d2->is_set()) || (d1->isa<Universe>() && d2->isa<Universe>())) return true;
    if (d1->sort() != d2->sort()) return false;

    // normalize: always put smaller gid to the left
    if (d1->gid() > d2->gid()) std::swap(d1, d2);

    // this assumption will either hold true - or we will bail out with false anyway
    auto [i, inserted] = equiv_.emplace(d1, d2);
    if (!inserted) return true;

    //if (!equiv(d1->type(), d2->type())) return false;

    if (d1->isa<Top>() || d2->isa<Top>()) return equiv(d1->type(), d2->type());

    if (is_sigma_or_arr(d1)) {
        if (!equiv(d1->arity(), d2->arity())) return false;

        if (auto a = isa_lit(d1->arity())) {
            for (size_t i = 0; i != a; ++i) {
                if (!equiv(proj(d1, *a, i), proj(d2, *a, i))) return false;
            }

            return true;
        }
    } else if (auto p1 = d1->isa<Param>()) {
        // params are equal if they appeared under the same binder
        for (auto [q1, q2] : params_) {
            if (p1 == q1) return d2->as<Param>() == q2;
        }
        return true;
    }

    //if (auto n1 = d1->isa_nominal())
        //params_.emplace_back(n1->param(), d2->as_nominal()->param());

    if (       d1->node   () != d2->node   ()
            || d1->fields () != d2->fields ()
            || d1->num_ops() != d2->num_ops()
            || d1->is_set () != d2->is_set()) return false;

    return std::equal(d1->ops().begin(), d1->ops().end(),
                      d2->ops().begin(), d2->ops().end(),
                      [&](auto op1, auto op2) { return equiv(op1, op2); });
}

bool Checker::assignable(const Def* type, const Def* val) {
    if (auto sigma = type->isa<Sigma>()) {
        if (!equiv(type->arity(), val->type()->arity())) return false;

        auto red = sigma->apply(val);
        for (size_t i = 0, e = red.size(); i != e; ++i) {
            if (!assignable(red[i], val->out(i))) return false;
        }

        return true;
    } else if (auto arr = type->isa<Arr>()) {
        if (!equiv(type->arity(), val->type()->arity())) return false;

        if (auto n = isa_lit(arr->arity())) {;
            for (size_t i = 0; i != *n; ++i) {
                if (!assignable(arr->apply(world_.lit_int(*n, i)).back(), val->out(i))) return false;
            }
        } else {
            return equiv(arr, val->type());
        }

        return true;
    } else {
        return equiv(type, val->type());
    }

    return false;
}

}
