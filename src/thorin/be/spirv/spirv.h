#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/be/spirv/spirv_builder.hpp"
#include "thorin/be/codegen.h"

namespace thorin::spirv {

using SpvId = builder::SpvId;

struct SpvType {
    SpvId id;
    size_t size = 0;

    // TODO: Alignment rules are complicated and client API dependant
    size_t alignment = 0;

    SpvId payload_id = { 0 };
};

struct FnBuilder;

struct BasicBlockBuilder : public builder::SpvBasicBlockBuilder {
    explicit BasicBlockBuilder(FnBuilder& fn_builder);

    std::unordered_map<const Param*, Phi> phis_map;
    DefMap<SpvId> args;
};

struct FnBuilder : public builder::SpvFnBuilder {
    const Scope* scope;
    builder::SpvFileBuilder* file_builder;
    std::vector<std::unique_ptr<BasicBlockBuilder>> bbs;
    std::unordered_map<Continuation*, BasicBlockBuilder*> bbs_map;
    ContinuationMap<SpvId> labels;
    DefMap<SpvId> params;
};

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".spv"; }
protected:
    void structure_loops();
    void structure_flow();

    SpvType convert(const Type*);
    void emit(const Scope& scope);
    void emit_epilogue(Continuation*, BasicBlockBuilder* bb);
    SpvId emit(const Def* def, BasicBlockBuilder* bb);

    SpvId get_codom_type(const Continuation* fn);

    builder::SpvFileBuilder* builder_ = nullptr;
    Continuation* entry_ = nullptr;
    FnBuilder* current_fn_ = nullptr;
    TypeMap<SpvType> types_;
    DefMap<SpvId> defs_;
};

}

#endif //THORIN_SPIRV_H
