#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/be/spirv/spirv_builder.hpp"
#include "thorin/analyses/schedule.h"
#include "thorin/be/codegen.h"
#include "thorin/be/emitter.h"

namespace thorin::spirv {

using SpvId = builder::SpvId;

class CodeGen;

struct FileBuilder;
struct FnBuilder;

struct SpvTargetInfo {
    struct {
        // Either '4' or '8'
        size_t pointer_size;
    } mem_layout;

    enum Dialect{
        OpenCL,
        Shady
    };
};

struct ConvertedType {
    SpvId id;
    struct Layout {
        size_t size, alignment;
    };
    std::optional<Layout> layout;
};

struct BasicBlockBuilder : public builder::SpvBasicBlockBuilder {
    explicit BasicBlockBuilder(FnBuilder& fn_builder);
    BasicBlockBuilder(const BasicBlockBuilder&) = delete;

    FnBuilder& fn_builder;
    FileBuilder& file_builder;
    std::unordered_map<const Param*, Phi> phis_map;
    DefMap<SpvId> args;
};

struct FnBuilder : public builder::SpvFnBuilder {
    explicit FnBuilder(FileBuilder& file_builder);
    FnBuilder(const FnBuilder&) = delete;

    FileBuilder& file_builder;
    std::vector<std::unique_ptr<BasicBlockBuilder>> bbs;
    DefMap<SpvId> params;
};

struct Builtins {
    SpvId workgroup_size;
    SpvId num_workgroups;
    SpvId workgroup_id;
    SpvId local_id;
    SpvId global_id;
    SpvId local_invocation_index;

    explicit Builtins(FileBuilder&);
};

struct ImportedInstructions {
    SpvId shader_printf;

    explicit ImportedInstructions(FileBuilder&);
};

struct FileBuilder : public builder::SpvFileBuilder {
    explicit FileBuilder(CodeGen* cg);
    FileBuilder(const FileBuilder&) = delete;

    CodeGen* cg;

    std::unique_ptr<Builtins> builtins;
    std::unique_ptr<ImportedInstructions> imported_instrs;

    SpvId u32_t();
    SpvId u32_constant(uint32_t);

private:
    SpvId u32_t_ { 0 };
};

class CodeGen : public thorin::CodeGen, public thorin::Emitter<SpvId, ConvertedType, BasicBlockBuilder*, CodeGen> {
public:
    CodeGen(Thorin& thorin, SpvTargetInfo, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".spv"; }

    bool is_valid(SpvId id) {
        return id > 0;
    }

    ConvertedType convert(const Type*);

    SpvId emit_fun_decl(Continuation*);

    FnBuilder* prepare(const Scope&);
    void prepare(Continuation*, FnBuilder*);
    void emit_epilogue(Continuation*);
    void finalize(const Scope&);
    void finalize(Continuation*);

    SpvId emit_constant(const Def*);
    SpvId emit_bb(BasicBlockBuilder* bb, const Def* def);
protected:
    FnBuilder& get_fn_builder(Continuation*);
    std::vector<SpvId> emit_builtin(const App&, const Continuation*, BasicBlockBuilder*);

    SpvId get_codom_type(const Continuation* fn);

    SpvTargetInfo target_info_;
    std::unique_ptr<FileBuilder> builder_;
    FnBuilder* current_fn_ = nullptr;
    ContinuationMap<std::unique_ptr<FnBuilder>> fn_builders_;
    const Cont2Config& kernel_config_;
};

}

#endif //THORIN_SPIRV_H
