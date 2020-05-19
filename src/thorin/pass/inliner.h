#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass<Inliner> {
public:
    Inliner(PassMan& man, size_t index)
        : Pass(man, index, "inliner")
    {}

    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    using State = std::tuple<LamSet>;

private:
    bool first_inline(Lam* lam) { auto [i, undo, ins] = put<LamSet>(lam); return ins; }
    std::optional<size_t> inlined_once(Lam* lam) { auto [i, undo, ins] = put<LamSet>(lam); return undo; }

    LamSet keep_;
};

}

#endif
