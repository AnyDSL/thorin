#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Performs β- and η-reduction.
 * * β-reduction happens optimistically if a @p Lam only occurs exactly once in callee postion.
 * * η-reduction: (TODO)
 *     * λx.f x -> f, whenever x does not appear free in f.
 *     * f -> λx.f x, if f occurs in both callee and some other position.
 */
class Reducer : public Pass<Reducer> {
public:
    Reducer(PassMan& man, size_t index)
        : Pass(man, index, "reducer")
    {}

    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    using Data = std::tuple<LamSet>;

private:
    bool is_candidate(Lam* lam) { return !ignore(lam) && !man().is_tainted(lam); }

    LamSet keep_;
};

}

#endif
