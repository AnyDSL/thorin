#ifndef THORIN_UNBOX_CLOSURE_H
#define THORIN_UNBOX_CLOSURE_H

#include <map>
#include <tuple>

#include "thorin/check.h"
#include "thorin/pass/pass.h"

namespace thorin {

class UnboxClosure : public FPPass<UnboxClosure, Lam> {
public:

    UnboxClosure(PassMan& man) 
        : FPPass<UnboxClosure, Lam>(man, "unbox_closures")
        , keep_(), boxed2unboxed_(), checker_(man.world())
    {}

    const Def* rewrite(const Def*) override;
    undo_t analyze(const Proxy*) override;

    using ArgSpec = std::map<size_t, const Def*>;
    using Data = LamMap<ArgSpec>;

private:
    DefSet keep_;
    LamMap<std::tuple<Lam*, DefVec>> boxed2unboxed_;
    Checker checker_;

    using Res = std::tuple<const Def*, const Def*, const Def*>;
    Res unbox(const Def*);


};

};

#endif
