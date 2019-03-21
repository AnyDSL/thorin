#ifndef THORIN_PASS_COPY_PROP_H
#define THORIN_PASS_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

class CopyProp : public Pass<CopyProp> {
public:
    CopyProp(PassMan& man, size_t id)
        : Pass(man, id)
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

        bool join(const App* app) {
            bool todo = false;
            for (size_t i = 0, e = params.size(); i != e; ++i) {
                auto& lattice = std::get<Lattice   >(params[i]);
                auto& param   = std::get<const Def*>(params[i]);

                if (lattice == Top || is_bot(app->arg(i))) continue;

                if (is_bot(param)) {
                    todo |= param != app->arg(i);
                    param = app->arg(i);
                } else if (param == app->arg(i)) {
                    /* do nothing */
                } else {
                    lattice = Top;
                    todo = true;
                }
            }
            return todo;
        }

        Array<std::tuple<Lattice, const Def*>> params;
        Lam* new_lam = nullptr;
        size_t undo;
    };

    using Lam2Info = DefMap<LamInfo>;
    using Lam2Lam  = LamMap<Lam*>;
    using State    = std::tuple<Lam2Info, Lam2Lam>;

private:
    auto& lam2info(Lam* lam) { return get<Lam2Info>(lam, LamInfo(lam, man().cur_state_id())); }
    auto& new2old(Lam* lam) { return get<Lam2Lam>  (lam); }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old(new_lam))
            return old_lam;
        return new_lam;
    }

    bool bot(Lam* lam) {
        bool todo = false;
        auto& info = lam2info(lam);
        info.new_lam = nullptr;
        for (auto& param : info.params) {
            todo |= std::get<Lattice>(param) != Top;
            std::get<Lattice>(param) = Top;
        }
        return todo;
    }
};

}

#endif
