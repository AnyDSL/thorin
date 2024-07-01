#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/analyses/schedule.h"
#include "thorin/be/codegen.h"
#include "thorin/be/emitter.h"

namespace thorin::spirv {

using SpvId = uint32_t;

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

struct Builtins {
    SpvId workgroup_size;
    SpvId num_workgroups;
    SpvId workgroup_id;
    SpvId local_id;
    SpvId global_id;
    SpvId local_invocation_index;

    explicit Builtins(FileBuilder&);
};

struct BasicBlockBuilder;

class CodeGen : public thorin::CodeGen, public thorin::Emitter<SpvId, ConvertedType, BasicBlockBuilder*, CodeGen> {
public:
    CodeGen(Thorin& thorin, SpvTargetInfo, bool debug, const Cont2Config* = nullptr);

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

    SpvId emit_as_bb(Continuation*);
    SpvId emit_mathop(BasicBlockBuilder* bb, const MathOp& op);

    SpvId get_codom_type(const Continuation* fn);

    SpvTargetInfo target_info_;
    FileBuilder* builder_;
    const Cont2Config* kernel_config_;
};

}

#endif //THORIN_SPIRV_H
