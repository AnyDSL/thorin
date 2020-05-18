#ifndef THORIN_PASS_COPY_PROP_H
#define THORIN_PASS_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * This one is similar to sparse conditional constant propagation (SCCP) but also propagates arbitrary values through Param%s.
 * The only crucial property is that the value to be propagated must be dominated by its parameter.
 * This is not necessarily the case in loops.
 * More precisely, "dominated" in this context means the value must not depend on its param.
 * Furthermore, this optmization also works on all Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
 */
class CopyProp : public Pass<CopyProp> {
public:
    CopyProp(PassMan& man, size_t index)
        : Pass(man, index, "copy_prop")
    {}

    void visit(Def*, Def*) override;
    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    struct Info {
        Lam* new_lam = nullptr;
        Array<const Def*> args;
    };

    using State = std::tuple<LamMap<Info>>;

private:
    bool join(const Def*& src, const Def* with);
    //Lam* original(Lam* new_lam) {
        //if (auto old_lam = new2old_.lookup(new_lam)) return *old_lam;
        //return new_lam;
    //}

    DefSet keep_;
};

}

#endif
