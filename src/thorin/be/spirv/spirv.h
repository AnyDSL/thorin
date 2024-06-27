#ifndef THORIN_SPIRV_H
#define THORIN_SPIRV_H

#include "thorin/be/spirv/spirv_builder.hpp"
#include "thorin/be/codegen.h"

namespace thorin::spirv {

using SpvId = builder::SpvId;

class CodeGen;
struct Datatype;
struct PtrDatatype;

struct FileBuilder;
struct FnBuilder;

struct ConvertedType {
    ConvertedType(CodeGen* cg) : code_gen(cg) {}
    ConvertedType(const ConvertedType&) = delete;

    spirv::CodeGen* code_gen;
    const thorin::Type* src_type;
    SpvId type_id { 0 };
    std::unique_ptr<Datatype> datatype;

    bool is_known_size() { return datatype != nullptr; }
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
    explicit FnBuilder(CodeGen* cg, FileBuilder& file_builder);
    FnBuilder(const FnBuilder&) = delete;

    CodeGen* cg;
    FileBuilder& file_builder;

    const Scope* scope = nullptr;
    std::vector<std::unique_ptr<BasicBlockBuilder>> bbs;
    std::unordered_map<Continuation*, BasicBlockBuilder*> bbs_map;
    ContinuationMap<SpvId> labels;
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
    /*SpvId i32_t;
    SpvId u32_t;
    SpvId i64_t;
    SpvId u64_t;
    SpvId i32_constant(int32_t);
    SpvId i64_constant(int64_t);
    SpvId u64_constant(uint64_t);*/
};

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World&, Cont2Config&, bool debug);

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".spv"; }

    ConvertedType* convert(const Type*);
protected:
    void structure_loops();
    void structure_flow();

    void emit(const Scope& scope);
    void emit_epilogue(Continuation*, BasicBlockBuilder* bb);
    SpvId emit(const Def* def, BasicBlockBuilder* bb);
    std::vector<SpvId> emit_builtin(const Continuation*, const Continuation*, BasicBlockBuilder*);

    SpvId get_codom_type(const Continuation* fn);

    std::unique_ptr<FileBuilder> builder_;
    Continuation* entry_ = nullptr;
    FnBuilder* current_fn_ = nullptr;
    DefMap<std::unique_ptr<ConvertedType>> types_;
    DefMap<SpvId> defs_;
    const Cont2Config& kernel_config_;

    friend PtrDatatype;
};

/// Thorin data types are mapped to SPIR-V in non-trivial ways, this interface is used by the emission code to abstract over
/// potentially different mappings, depending on the capabilities of the target platform. The serdes code deals with pointers
/// in arrays of unsigned 32 bit words, and is there to get around the limitation of not being able to bitcast pointers in the
/// logical addressing mode.
struct Datatype {
public:
    ConvertedType* type;
    Datatype(ConvertedType* type) : type(type) {}

    // Datatypes are serialized using a base element, for now it is hardcoded to use 32-bit scalar unsigned integers
    static constexpr size_t base_element_bitwidth = 32;
    static constexpr size_t base_element_bytes = base_element_bitwidth / 8;

    virtual size_t serialized_size() = 0;
    virtual SpvId emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) = 0;
    virtual void emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) = 0;
};

/// For scalar datatypes
struct ScalarDatatype : public Datatype {
    int type_tag;
    size_t size_in_bytes;
    size_t alignment;
    ScalarDatatype(ConvertedType* type, int type_tag, size_t size_in_bytes, size_t alignment_in_bytes);

    size_t serialized_size() override { return (size_in_bytes + 3) / 4; };
    SpvId emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) override;
    void emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) override;
};

struct PtrDatatype : public Datatype {
    static constexpr size_t bitwidth = 64;
    PtrDatatype(ConvertedType* type) : Datatype(type) {}

    size_t serialized_size() override { return bitwidth / 32; };
    SpvId emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) override;
    void emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) override;
};

struct DefiniteArrayDatatype : public Datatype {
    ConvertedType* element_type;
    size_t length;

    DefiniteArrayDatatype(ConvertedType* type, ConvertedType* element_type, size_t length);

    size_t serialized_size() override { return element_type->datatype->serialized_size(); };
    SpvId emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) override;
    void emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) override;
};

struct ProductDatatype : public Datatype {
    std::vector<ConvertedType*> elements_types;
    size_t total_size = 0;

    ProductDatatype(ConvertedType* type, const std::vector<ConvertedType*>&& elements_types);

    size_t serialized_size() override { return total_size; };
    SpvId emit_deserialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset) override;
    void emit_serialization(BasicBlockBuilder& bb, spv::StorageClass storage_class, SpvId array, SpvId base_offset, SpvId data) override;
};

}

#endif //THORIN_SPIRV_H
