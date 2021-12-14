#ifndef THORIN_CLOSURE_DESTRUCT_H
#define THORIN_CLOSURE_DESTRUCT_H

#include <set>
#include <functional>

#include "thorin/pass/pass.h"

namespace thorin {

// class PTG;

class ClosureDestruct : public FPPass<ClosureDestruct, Lam> {
public:
    ClosureDestruct(PassMan& man) 
        : FPPass<ClosureDestruct, Lam>(man, "closure_destruct")
        , escape_(), clos2dropped_()
    {}

    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

    using Data = int;

private:
    DefSet escape_;
    LamMap<std::pair<const Def*, Lam*>> clos2dropped_;

    bool is_esc(const Def* def);

    undo_t join(DefSet& defs, bool cond);
    undo_t join(const Def* def, bool cond);
};

}
#endif
