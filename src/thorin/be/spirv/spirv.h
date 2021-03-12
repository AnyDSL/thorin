#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/be/spirv/spirv_builder.hpp"
#include "thorin/be/backends.h"

#include "thorin/analyses/schedule.h"

namespace thorin::spirv {

using SpvId = builder::SpvId;

struct BasicBlockBuilder : public builder::SpvBasicBlockBuilder {
    explicit BasicBlockBuilder(builder::SpvFileBuilder& file_builder)
    : builder::SpvBasicBlockBuilder(file_builder)
    {}

    std::unordered_map<const Param*, Phi> phis;
    DefMap<SpvId> args;
};

struct FnBuilder : public builder::SpvFnBuilder {
    std::unordered_map<Continuation*, BasicBlockBuilder*> bbs_map;
    ContinuationMap<SpvId> labels;
    DefMap<SpvId> params;
};

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit(std::ostream& stream) override;
protected:
    void structure_loops();
    void structure_flow();

    SpvId convert(const Type*);
    void emit(const Scope& scope);
    void emit_epilogue(Continuation*, BasicBlockBuilder* bb);
    SpvId emit(const Def* def, BasicBlockBuilder* bb);

    SpvId get_codom_type(const Continuation* fn);

    builder::SpvFileBuilder* builder_ = nullptr;
    Continuation* entry_ = nullptr;
    FnBuilder* current_fn_ = nullptr;
    Scheduler scheduler_;
    TypeMap<SpvId> types_;
    DefMap<SpvId> defs_;
};

}

#endif //THORIN_SPIRV_H
