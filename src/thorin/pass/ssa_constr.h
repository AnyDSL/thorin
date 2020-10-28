#ifndef THORIN_PASS_SSA_CONSTR_H
#define THORIN_PASS_SSA_CONSTR_H

#include <map>
#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau.
 * We use the following lattice:
 * @code
 *                    keep
 *                   /    \
 *             preds_n     preds1_non_callee_pos
 *              \             /
 * preds_1_callee_pos        /
 *                  \       /
 *                   preds_0
 * @endcode
 */
class SSAConstr : public Pass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : Pass(man, index, "ssa_constr")
    {}

    struct Visit {
        bool callee_pos = true;
        Lam* pred = nullptr;
    };

    struct Enter {
        GIDSet<const Proxy*> writable;
    };

    using State = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    enum : flags_t { Sloxy, Phixy, Traxy };

    void enter(Def*) override;
    const Def* prewrite(Def*, const Def*);
    std::variant<const Def*, undo_t> rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    undo_t join(Lam* cur_lam, Lam* lam, bool callee_pos);
    std::variant<const Def*, undo_t> mem2phi(Lam*, const App*, Lam*);

    template<class T> // T = Visit or Enter
    std::tuple<T&, undo_t, bool> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<T>>(lam); return {i->second, undo, ins}; }
    bool keep(Lam* lam) { return lam->is_external() || !lam->is_set() || keep_.contains(lam); }

    size_t slot_id_;
    std::map<Lam*, GIDMap<const Proxy*, const Def*>, GIDLt<Lam*>> lam2sloxy2val_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phis_; ///< Contains the phis we have to add to the mem_lam to build the phi_lam.
    DefSet keep_;                                                  ///< Contains Lams as well as sloxys we want to keep.
    LamSet preds_n_;                                               ///< Contains Lams with more than one preds.
    Lam2Lam mem2phi_;
};

}

#endif
