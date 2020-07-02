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

    using Args  = std::vector<const Def*>;
    using State = std::tuple<LamMap<Args>>;

private:
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    std::pair<Args&, undo_t> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<Args>>(lam); return {i->second, undo}; }

    LamMap<Lam*> param2prop_;
    DefSet keep_;
};

}

#endif
