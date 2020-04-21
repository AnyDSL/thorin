#ifndef THORIN_PASS_COPY_PROP_H
#define THORIN_PASS_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * This one is similar to constant propagation but also propagates arbitrary values through phis.
 * The only crucial property is that the value to be propagated must be dominated by its parameter.
 * This is not necessarily the case in loops.
 * More precisely, "dominated" in this context means the value must not depend on its phi.
 */
class CopyProp : public Pass {
public:
    CopyProp(PassMan& man, size_t index)
        : Pass(man, index, "copy_prop")
    {}

    bool enter(Def* def) override { return def->isa<Lam>(); }
    Def* inspect(Def*) override;
    const Def* rewrite(const Def*) override;
    bool analyze(const Def*) override;

    struct Info {
        Lam* new_lam = nullptr;
        Array<const Def*> args;
    };

private:
    bool join(const Def*& src, const Def* with);
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old_.lookup(new_lam)) return *old_lam;
        return new_lam;
    }

    LamMap<Info> lam2info_;
    LamMap<Lam*> new2old_;
    DefSet keep_;
};

}

#endif
