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

    // 3 cases:
    // * lam is keep_                        -> never inline
    // * lam is not in this map              -> we have never seen this and it's safe to inline
    // * lam is in this map but not in keep_ -> inlined once
    using State = std::tuple<LamSet>;

private:
    bool first_inline(Lam* lam) { return std::get<2>(put<LamSet>(lam)); }
    std::optional<size_t> inlined_once(Lam* lam) { return contains<LamSet>(lam); }

    LamSet keep_;
};

}

#endif
