#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau
 */
class Mem2Reg : public Pass<Mem2Reg> {
public:
    Mem2Reg(PassMan& man, size_t index)
        : Pass(man, index)
    {}

    const Def* rewrite(const Def*) override;
    void inspect(Def*) override;
    void enter(Def*) override;
    void analyze(const Def*) override;

    struct Info {
        enum Lattice { Preds0, Preds1, PredsN, Keep };

        Info() = default;
        Info(Info&& other) { swap(*this, other); }
        Info(const Info& info)
            : proxy2val(info.proxy2val)
            , writable (info.writable)
            , pred     (info.pred)
            , new_lam  (info.new_lam)
            , num_slots(info.num_slots)
            , lattice  (info.lattice)
            , undo     (info.undo)
        {}
        Info(size_t undo)
            : lattice(Preds0)
            , undo(undo)
        {}

        Info& operator=(Info other) { swap(*this, other); return *this; }

        friend void swap(Info& a, Info& b) {
            using std::swap;
            swap(a.proxy2val,    b.proxy2val);
            swap(a.writable,     b.writable);
            swap(a.pred,         b.pred);
            swap(a.new_lam,      b.new_lam);
            swap(a.num_slots,    b.num_slots);
            swap(a.lattice_undo, b.lattice_undo);
        }

        GIDMap<const Analyze*, const Def*> proxy2val;
        GIDSet<const Analyze*> writable;
        Lam* pred = nullptr;
        Lam* new_lam = nullptr;
        unsigned num_slots = 0;
        union {
            struct {
                unsigned lattice :  2;
                unsigned undo    : 30;
            };
            unsigned lattice_undo;
        };
    };

    using Lam2Info = LamMap<Info>;
    using State    = std::tuple<Lam2Info>;

private:
    const Analyze* isa_proxy(const Def*);
    const Analyze* isa_virtual_phi(const Def*);
    const Def* get_val(Lam*, const Analyze*);
    const Def* get_val(const Analyze* proxy) { return get_val(man().cur_lam(), proxy); }
    const Def* set_val(Lam*, const Analyze*, const Def*);
    const Def* set_val(const Analyze* proxy, const Def* val) { return set_val(man().cur_lam(), proxy, val); }

    auto& lam2info(Lam* lam) { return get<Lam2Info>(lam, Info(man().cur_state_id())); }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old_.lookup(new_lam)) return *old_lam;
        return new_lam;
    }

    LamMap<Lam*> new2old_;
    LamMap<std::set<const Analyze*, GIDLt<const Analyze*>>> lam2phis_;
    DefSet keep_;
    LamSet preds_n_;
};

}

#endif
