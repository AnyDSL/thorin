#ifndef THORIN_CLOSURE_CONV_H
#define THORIN_CLOSURE_CONV_H

#include <memory>
#include <optional>

#include "thorin/pass/pass.h"

namespace thorin  {

class ClosureConv : public RWPass {
public:
    ClosureConv(PassMan &man)
        : RWPass(man, "ClosureConv")
        , status_()
        , old_params_()
        , cur_old_param_(nullptr)
        , rewrite_cur_nom_(true)
    {}

    void enter() override;
    const Def* rewrite(Def* nom, const Def* typ, const Def* dbg) override;
    const Def* rewrite(const Def* def, const Def* type, Defs ops, const Def* dbg) override;

    void finish() override {
        if (auto cur_lam = cur_nom<Lam>()) {
            world().DLOG("finished closure stub {}", cur_lam);
            cur_lam->dump(90000);
            cur_lam->type()->dump(90000);
            world().debug_stream();
        }
    }

// private:

    enum Status { 
        UNPROC,
        CL_STUB,
        DONE
    };

    DefMap<Status> status_;
    Def2Def old_params_;
    const Def *cur_old_param_;
    bool rewrite_cur_nom_;

    Status status(const Def* def) {
        if (auto stat = status_.lookup(def))
            return *stat;
        else
            return UNPROC;
    }

    void mark(const Def* def, Status status) {
        status_.emplace(def, status);
    }

    bool should_rewrite(const Def *def) {
        auto b = rewrite_cur_nom_ && status(def) == UNPROC;
        if (!b) {
            world().DLOG("--> Skip rewrite");
        }
        return b;
    }

    const Def* old_param() {
        return cur_old_param_;
    }

    void push_old_param(Lam* old, Lam* lifted) {
        old_params_.emplace(lifted, old->var());
    }


    const Def* rewrite_rec(const Def* def);

    template<bool rewrite_args>
    const Pi* lifted_fn_type(const Pi* pi, const Def *env_type);

    template<bool rewrite_args>
    Sigma* closure_type(const Pi* pi);

    const Def* closure_stub(Lam *lam);
};

}

#endif
