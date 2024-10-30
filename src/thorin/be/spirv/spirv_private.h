#ifndef THORIN_SPIRV_PRIVATE_H
#define THORIN_SPIRV_PRIVATE_H

#include "spirv.h"

#include "thorin/be/spirv/spirv_builder.hpp"

namespace thorin::spirv {

struct BasicBlockBuilder : public builder::SpvBasicBlockBuilder {
    explicit BasicBlockBuilder(FnBuilder& fn_builder);

    BasicBlockBuilder(const BasicBlockBuilder&) = delete;

    FnBuilder& fn_builder;
    FileBuilder& file_builder;
    std::unordered_map<const Param*, Phi> phis_map;

    bool semi_inline;
};

struct FnBuilder : public builder::SpvFnBuilder {
    explicit FnBuilder(FileBuilder& file_builder);

    FnBuilder(const FnBuilder&) = delete;

    FileBuilder& file_builder;
    std::vector<std::unique_ptr<BasicBlockBuilder>> bbs;
    DefMap<SpvId> params;
};

struct FileBuilder : public builder::SpvFileBuilder {
    explicit FileBuilder(CodeGen* cg);
    FileBuilder(const FileBuilder&) = delete;

    CodeGen* cg;

    FnBuilder* current_fn_ = nullptr;
    ContinuationMap<std::unique_ptr<FnBuilder>> fn_builders_;
    std::unordered_map<uint32_t, SpvId> builtins_;
    std::vector<SpvId> interface;

    SpvId u32_t();
    SpvId u32_constant(uint32_t);

private:
    SpvId u32_t_ { 0 };
};

}

#endif // THORIN_SPIRV_PRIVATE_H
