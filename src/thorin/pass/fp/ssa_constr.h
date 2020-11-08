#ifndef THORIN_PASS_FP_SSA_CONSTR_H
#define THORIN_PASS_FP_SSA_CONSTR_H

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
 *             Top                      <-+
 *           /      \                     |- Glob(al lattice)
 *        PredsN     \                  <-+
 *           \        \
 * Preds1_Callee      Preds1_Non_Callee <--- Loc(al lattice)
 *             \      /
 *              Preds0                  <--- not in any map
 * @endcode
 */
class SSAConstr : public FPPass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : FPPass(man, index, "ssa_constr")
    {}

    enum class Loc  : bool { Preds1_Callee, Preds1_Non_Callee };
    enum class Glob : bool { PredsN, Top };
    enum : flags_t { Sloxy, Phixy, Traxy };

    struct Visit {
        Loc loc;
        Lam* pred = nullptr;
    };

    struct Enter {
        GIDSet<const Proxy*> writable;
    };

    using Data = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;
    virtual undo_t analyze(Def* cur_nom) override;
    undo_t analyze(Def*, const Def*) override;

    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    undo_t join(Lam* cur_lam, Lam* lam, Loc);
    const Def* mem2phi(Lam*, const App*, Lam*);

    size_t slot_id_;
    std::map<Lam*, GIDMap<const Proxy*, const Def*>, GIDLt<Lam*>> lam2sloxy2val_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phixys_; ///< Contains the @p Phixy%s to add to @c mem_lam to build the @c phi_lam.
    GIDSet<const Proxy*> keep_;                                      ///< Contains @p Sloxy%s we want to keep.
    LamMap<Glob> lam2glob_;
    Lam2Lam mem2phi_;
    DefSet analyzed_;
};

}

#endif
