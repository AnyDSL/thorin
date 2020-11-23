#ifndef THORIN_PASS_RW_COMPILE_PTRNS_H
#define THORIN_PASS_RW_COMPILE_PTRNS_H

#include "thorin/pass/pass.h"

namespace thorin {

/// Pattern matching compiler based on:
/// "Compiling Pattern Matching to Good Decision Trees"
/// by Luc Maranget
class PtrnCompiler : public RWPass {
public:
    PtrnCompiler(PassMan& man)
        : RWPass(man, "ptrn_copmiler")
    {}

    const Def* rewrite(Def*, const Def*) override;

private:
    const Def* compile(const Match* match, const Def* arg, std::vector<Ptrn*>& ptrns, const Def* dbg);
    const Def* compile(const Match* match);

    /// Flattens the tuples/packs in a pattern
    Ptrn* flatten(Ptrn* ptrn);

    /// Eliminates an element from a tuple
    std::pair<const Def*, const Def*> eliminate(const Def* def, size_t col);

    /// Introduces an element in a tuple
    const Def* introduce(const Def* def, size_t col, const Def* val);

    /// Specializes the pattern for one particular value
    Ptrn* specialize(Ptrn* ptrn, size_t col, const Def* ctor, const Def* arg_col, const Def* s_arg, bool d_arg_was_empty);

    /// Returns whether the constructor patterns form a signature for the matched type
    bool is_complete(const Def* arg_type, const DefMap<std::vector<Ptrn*>>& ctor2ptrns);

    DefMap<bool> redundant_;
    Def2Def parent_;
};


}

#endif
