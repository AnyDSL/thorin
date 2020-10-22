#ifndef THORIN_PASS_SSA_CONSTR_H
#define THORIN_PASS_SSA_CONSTR_H

#include <map>
#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/// SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
/// This is loosely based upon:
/// "Simple and Efficient Construction of Static Single Assignment Form"
/// by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau.
class SSAConstr : public Pass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : Pass(man, index, "ssa_constr")
    {}

    struct Visit {
        Lam* phi_lam = nullptr;
    };

    struct Enter {
        GIDSet<const Proxy*> writable;
    };

    using State = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    enum : flags_t { Sloxy, Phixy, Traxy, Stoxy };

    //@{
    /// @name Proxy%s & helpers
    /// This thing replaces a @p slot within @p lam and identifies it via the @p slot_id_.
    const Proxy* make_sloxy(Lam* lam, const Def* slot);

    /// A virtual phi; marks the necessity to introduce a phi param within @p lam for @p sloxy.
    const Proxy* make_phixy(const Proxy* sloxy, Lam* lam);

    /// Tracks the pred @p lam for a @p mem value.
    const Proxy* make_traxy(const Def* mem, Lam* lam);
    //@}

    Def* mem2phi(Def*);
    void enter(Def*) override;
    const Def* prewrite(Def*, const Def*) override;
    std::variant<const Def*, undo_t> rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    std::variant<const Def*, undo_t> rewrite(Lam*, const App*, Lam*);

    template<class T> // T = Visit or Enter
    std::tuple<T&, undo_t, bool> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<T>>(lam); return {i->second, undo, ins}; }
    Lam* phi2mem(Lam* phi_lam) { auto mem_lam = phi2mem_.lookup(phi_lam); return mem_lam ? *mem_lam : nullptr; }
    Lam* lam2mem(Lam* lam) { auto mem_lam = phi2mem(lam); return mem_lam ? mem_lam : lam; }

    Lam* pred_ = nullptr;
    size_t slot_id_;
    std::map<Lam*, GIDMap<const Proxy*, const Def*>, GIDLt<Lam*>> lam2sloxy2val_;
    LamMap<Lam*> phi2mem_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phis_; ///< Contains the phis we have to add to the mem_lam to build the phi_lam.
    DefSet keep_;                                                  ///< Contains Lams as well as sloxys we want to keep.
    LamSet preds_n_;                                               ///< Contains Lams with more than one preds.
};

}

#endif
