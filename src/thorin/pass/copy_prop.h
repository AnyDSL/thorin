#ifndef THORIN_PASS_COPY_PROP_H
#define THORIN_PASS_COPY_PROP_H

#if 0
#include "thorin/pass/pass.h"

namespace thorin {

class CopyProp : public Pass<CopyProp> {
public:
    CopyProp(PassMan& man, size_t index)
        : Pass(man, index)
    {}

    const Def* rewrite(const Def*) override;
    void inspect(Def*) override;
    void enter(Def*) override;
    void analyze(const Def*) override;

    enum Lattice { Val, Top };

    struct LamInfo {
        LamInfo() = default;
        LamInfo(Lam* lam, size_t undo)
            : params(lam->num_params(), [&](auto i) { return std::tuple(Val, lam->world().bot(lam->domain(i))); })
            , undo(undo)
        {}

        bool join(const App*);

        Array<std::tuple<Lattice, const Def*>> params;
        Lam* new_lam = nullptr;
        size_t undo;
    };

    using Lam2Info = DefMap<LamInfo>;
    using Lam2Lam  = LamMap<Lam*>;
    using State    = std::tuple<Lam2Info, Lam2Lam>;

private:
    bool set_top(Lam*);

    auto& lam2info(Lam* lam) { return get<Lam2Info>(lam, LamInfo(lam, man().cur_state_id())); }
    auto& new2old(Lam* lam) { return get<Lam2Lam>  (lam); }
};

}

#endif
#endif
