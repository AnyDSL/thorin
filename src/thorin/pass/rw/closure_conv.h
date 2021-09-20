#ifndef THORIN_CLOSURE_CONV_H
#define THORIN_CLOSURE_CONV_H

#include "thorin/pass/pass.h"

namespace thorin  {

class ClosureConv : public RWPass {
public:
    ClosureConv(PassMan &man)
        : RWPass(man, "ClosureConv")
        , status_()
        , fv_maps_()
        , cur_fv_map_(nullptr)
        , rewrite_cur_nom_(false)
    {}

    void enter() override;
    const Def* rewrite(Def* nom, const Def* typ, const Def* dbg) override;
    const Def* rewrite(const Def* def, const Def* type, Defs ops, const Def* dbg) override;

private:

    class FVMap {
    private:
        Def2Def map_;
        const Def* old_param_;
    public:
        FVMap(const Def *old_param)
            : map_(DefMap<const Def*>())
            , old_param_(old_param)
        {}

        std::optional<const Def*> lookup(const Def* fv) {
            return map_.lookup(fv);
        }

        void emplace(const Def* old_def, const Def* new_def) {
            map_.emplace(old_def, new_def);
        }

        const Def* old_param() { return old_param_; }
    };

    using FVMapPtr = std::unique_ptr<FVMap>;

    enum Status { 
        UNPROC,
        CL_STUB,
        DONE
    };

    DefMap<Status> status_;
    DefMap<FVMapPtr> fv_maps_;
    FVMapPtr cur_fv_map_;
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
        return rewrite_cur_nom_ && status(def) == UNPROC;
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
