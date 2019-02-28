#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass<Inliner> {
public:
    Inliner(PassMgr& mgr, size_t pass_index)
        : Pass(mgr, pass_index)
    {}

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

    enum Lattice { Bottom, Inlined_Once, Dont_Inline };

    struct Info {
        Info() = default;
        Info(Lattice lattice, size_t undo)
            : lattice(lattice)
            , undo(undo)
        {}

        unsigned lattice :  4;
        unsigned undo    : 28;
    };

    using State = std::tuple<LamMap<Info>>;

private:
    Info& info(Lam* lam) { return get<LamMap<Info>>(lam, Info(Lattice::Bottom, mgr().state_id())); }
};

}

#endif
