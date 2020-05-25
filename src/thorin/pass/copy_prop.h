#ifndef THORIN_PASS_COPY_PROP_H
#define THORIN_PASS_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

/// This one is similar to sparse conditional constant propagation (SCCP) but also propagates arbitrary values through Param%s.
/// However, this optmization also works on all @p Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
/// What is more, this optimization will also propagate arbitrary @p Def%s.
class CopyProp : public Pass<CopyProp> {
public:
    CopyProp(PassMan& man, size_t index)
        : Pass(man, index, "copy_prop")
    {}

    void visit(Def*, Def*) override;
    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    struct Visit {
        Lam* prop_lam = nullptr;
    };

    struct Enter {
    };

    using State = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    template<class T> // T = Visit or Enter
    std::pair<T&, undo_t> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<T>>(lam); return {i->second, undo}; }
    Lam* prop2param(Lam* prop_lam) { auto param_lam = prop2param_.lookup(prop_lam); return param_lam ? *param_lam : nullptr; }
    Lam* lam2param(Lam* lam) { auto param_lam = prop2param(lam); return param_lam ? param_lam : lam; }

    LamMap<std::vector<const Def*>> args_;
    LamMap<Lam*> prop2param_;
    DefSet keep_;
};

}

#endif
