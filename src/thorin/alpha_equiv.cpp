#include "thorin/def.h"

namespace thorin {

struct Checker {
    using Pair = std::tuple<const Def*, const Def*>;

    bool run(const Def* d1, const Def* d2) {
        if (d1 == d2 || (!d1->is_set() && !d2->is_set())) return true;
        if (d1->node() != d2->node() || d1->fields() != d2->fields() || d1->num_ops() != d2->num_ops()) return false;
        if (d1->gid() > d2->gid()) std::swap(d1, d2); // normalize: always put smaller gid to the left
        if (equiv.contains({d1, d2})) return true;

        bool result = true;
        if (auto p1 = d1->isa<Param>()) {
            auto p2 = d2->as<Param>();
            for (auto [q1, q2] : params) {
                if (p1 == q1) return p2 == q2;
            }
        } else if (auto n1 = d1->isa_nominal()) {
            if (auto n2 = d2->isa_nominal()) {
                params.emplace_back(n1->param(), n2->param());
                equiv.emplace(n1, n2);
                for (size_t i = 0, e = n1->num_ops(); i != e && result; ++i)
                    result &= run(n1->op(i), n2->op(i));
            } else {
                return false;
            }
        } else {
            if (!d2->isa_nominal()) {
                for (size_t i = 0, e = d1->num_ops(); i != e && result; ++i)
                    result &= run(d1->op(i), d2->op(i));

                if (result)
                    equiv.emplace(d1, d2);
            } else {
                return false;
            }
        }

        return result;
    }

    struct Hash {
        static uint32_t hash(Pair pair) { return hash_combine(hash_begin(std::get<0>(pair)), std::get<1>(pair)); }
        static bool eq(Pair p1, Pair p2) { return p1 == p2; }
        static Pair sentinel() { return {nullptr, nullptr}; }
    };

    HashSet<Pair, Hash> equiv;
    std::deque<std::tuple<const Param*, const Param*>> params;
};

bool alpha_equiv(const Def* d1, const Def* d2) { return Checker().run(d1, d2); }

}
