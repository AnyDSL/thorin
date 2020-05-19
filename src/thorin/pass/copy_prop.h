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
        std::vector<const Def*> args;
    };

    struct Enter {
    };

    using State = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    template<class T>
    std::pair<T&, undo_t> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<T>>(lam); return {i->second, undo}; }

    Lam* param2prop(Lam* param_lam) { auto  prop_lam = param2prop_.lookup(param_lam); return  prop_lam ? * prop_lam : nullptr; }
    Lam* prop2param(Lam*  prop_lam) { auto param_lam = prop2param_.lookup( prop_lam); return param_lam ? *param_lam : nullptr; }
    Lam* param2lam(Lam* lam) { auto prop_lam = param2prop(lam); return prop_lam ? prop_lam : lam; }

    LamMap<Lam*> param2prop_;
    LamMap<Lam*> prop2param_;
    DefSet keep_;
};

}

#endif
