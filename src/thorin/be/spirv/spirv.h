#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/analyses/schedule.h"
#include "thorin/be/codegen.h"
#include "thorin/be/emitter.h"

namespace thorin::spirv {

using Id = uint32_t;

class CodeGen;

struct FileBuilder;
struct FnBuilder;

struct Target {
    struct {
        // Either '4' or '8'
        size_t pointer_size = 8;
    } mem_layout;

    struct {
        bool broken_op_construct_composite = true;
        bool static_ac_indices_must_be_i32 = true;
    } bugs;

    enum Dialect {
        OpenCL,
        Vulkan
    };

    Dialect dialect = OpenCL;
};

struct ConvertedType {
    Id id;
    struct Layout {
        size_t size, alignment;
    };
    std::optional<Layout> layout;
    struct {
        std::optional<const thorin::Type*> payload_t;
    } variant;
};

struct BasicBlockBuilder;

class CodeGen : public thorin::CodeGen, public thorin::Emitter<Id, ConvertedType, BasicBlockBuilder*, CodeGen> {
public:
    CodeGen(Thorin& thorin, Target&, bool debug, const Cont2Config* = nullptr);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".spv"; }

    bool is_valid(Id id) {
        return id > 0;
    }

    uint32_t convert(AddrSpace);
    ConvertedType convert(const Type*, bool allow_void = false);

    Id emit_fun_decl(Continuation*);

    FnBuilder* prepare(const Scope&);
    void prepare(Continuation*, FnBuilder*);
    void emit_epilogue(Continuation*);
    void finalize(const Scope&);
    void finalize(Continuation*);

    Id emit_constant(const Def*);
    Id emit_bb(BasicBlockBuilder* bb, const Def* def);
protected:
    FnBuilder& get_fn_builder(Continuation*);
    std::vector<Id> emit_intrinsic(const App& app, const Continuation* intrinsic, BasicBlockBuilder* bb);
    std::vector<Id> emit_args(Defs);
    Id literal(uint32_t);

    Id emit_as_bb(Continuation*);
    Id emit_mathop(BasicBlockBuilder* bb, const MathOp& op);
    Id emit_composite(BasicBlockBuilder* bb, Id, Defs);
    Id emit_composite(BasicBlockBuilder* bb, Id, ArrayRef<Id>);
    Id emit_ptr_bitcast(BasicBlockBuilder* bb, const PtrType* from, const PtrType* to, Id);

    std::tuple<std::vector<Id>, Id> get_dom_codom(const FnType* fn);
    Id get_codom_type(const FnType*);

    Target target_info_;
    FileBuilder* builder_;
    const Cont2Config* kernel_config_;

    friend Target;
};

}

#endif //THORIN_SPIRV_H
