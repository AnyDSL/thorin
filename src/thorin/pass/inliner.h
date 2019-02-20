#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass {
public:
    Inliner(PassMgr& mgr)
        : Pass(mgr)
    {}

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

private:
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

    static_assert(sizeof(Info) == 4);

    Info& info(Lam* lam) { return info_.emplace(lam, Info(Lattice::Bottom, mgr().num_states())).first->second;  }

    // TODO we must also undo this info
    LamMap<Info> info_;
};

}

#endif
