#include "thorin/def.h"

namespace thorin {

struct Checker {
    bool run(const Def* d1, const Def* d2) {
        if (d1 == d2 || (!d1->is_set() && !d2->is_set())) return true;
        if (d1->node() != d2->node() || d1->fields() != d2->fields() || d1->num_ops() != d2->num_ops()
                || bool(d1->isa_nominal()) != bool(d2->isa_nominal()) || d1->is_set() != d2->is_set()) return false;
        if (d1->gid() > d2->gid()) std::swap(d1, d2); // normalize: always put smaller gid to the left

        // this assumption will either hold true - or we will bail out with false anyway
        auto [i, success] = equiv.emplace(d1, d2);
        if (!success) return true;

        // params are equal if they appeared under the same binder
        if (auto p1 = d1->isa<Param>()) {
            for (auto [q1, q2] : params) {
                if (p1 == q1) return d2->as<Param>() == q2;
            }
            return true;
        }

        if (auto n1 = d1->isa_nominal())
            params.emplace_back(n1->param(), d2->as_nominal()->param());

        return std::equal(d1->ops().begin(), d1->ops().end(),
                          d2->ops().begin(), d2->ops().end(), [&](auto op1, auto op2) { return run(op1, op2); });
    }

    HashSet<DefDef, DefDefHash> equiv;
    std::deque<DefDef> params;
};

bool alpha_equiv(const Def* d1, const Def* d2) { return Checker().run(d1, d2); }

}
