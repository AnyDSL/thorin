#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau
 */
class Mem2Reg : public Pass<Mem2Reg> {
public:
    Mem2Reg(PassMan& man, size_t id)
        : Pass(man, id)
    {}

    const Def* rewrite(const Def*) override;
    void inspect(Def*) override;
    void enter(Def*) override;
    void reenter(Def*) override;
    void analyze(const Def*) override;

    struct ProxyInfo {
        enum Lattice { SSA, Keep };

        //ProxyInfo() = default;
        ProxyInfo()
            : lattice(SSA)
            , undo(0x7fffffff)
        {}
        ProxyInfo(size_t undo)
            : lattice(SSA)
            , undo(undo)
        {}

        unsigned lattice :  1;
        unsigned undo    : 31;
    };

    struct LamInfo {
        enum Lattice { Preds0, Preds1, PredsN, Keep };

        LamInfo() = default;
        LamInfo(size_t undo)
            : lattice(Preds0)
            , undo(undo)
        {}

        std::vector<const Analyze*> proxies;
        GIDMap<const Analyze*, const Def*> proxy2val;
        GIDSet<const Analyze*> writable;
        Lam* pred = nullptr;
        Lam* new_lam = nullptr;
        unsigned num_slots = 0;
        unsigned lattice    :  2;
        unsigned undo       : 30;
    };

    using Proxy2Info = GIDMap<const Analyze*, ProxyInfo>;
    using Lam2Info = LamMap<LamInfo>;
    using Lam2Lam  = LamMap<Lam*>;
    using State    = std::tuple<Proxy2Info, Lam2Info, Lam2Lam>;

private:
    const Analyze* isa_proxy(const Def*);
    const Def* get_val(Lam*, const Analyze*);
    const Def* get_val(const Analyze* proxy) { return get_val(man().cur_lam(), proxy); }
    const Def* set_val(Lam*, const Analyze*, const Def*);
    const Def* set_val(const Analyze* proxy, const Def* val) { return set_val(man().cur_lam(), proxy, val); }

    auto& proxy2info(const Analyze* proxy, size_t u) { return get<Proxy2Info>(proxy, ProxyInfo(u)); }
    auto& proxy2info(const Analyze* proxy) { return get<Proxy2Info>(proxy); }
    auto& lam2info  (Lam* lam)             { return get<Lam2Info>(lam, LamInfo(man().cur_state_id())); }
    auto& new2old   (Lam* lam)             { return get<Lam2Lam> (lam); }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old(new_lam)) return old_lam;
        return new_lam;
    }
};

}

#endif
