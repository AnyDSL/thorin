#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass<Inliner> {
public:
    Inliner(PassMan& man, size_t index)
        : Pass(man, index, "inliner")
    {}

    std::variant<const Def*, undo_t> rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    using State = std::tuple<LamSet>;

private:
    bool is_candidate(Lam* lam) { return lam != nullptr && lam->is_set() && !lam->is_external() && !lam->is_intrinsic() && !man().is_tainted(lam); }

    LamSet keep_;
};

}

#endif
