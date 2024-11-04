#include <spirv/unified1/spirv.hpp>

#include <string>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <ostream>
#include <memory>

namespace thorin::spirv::builder {

//struct SpvId { uint32_t id; };
using SpvId = uint32_t;

struct SpvSectionBuilder;
struct SpvBasicBlockBuilder;
struct SpvFnBuilder;
struct SpvFileBuilder;

struct ExtendedInstruction {
    const char* set_name;
    uint32_t id;
};

inline int div_roundup(int a, int b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

struct SpvSectionBuilder {
    std::vector<uint32_t> data_;

private:
    void output_word(uint32_t word) {
        data_.push_back(word);
    }
public:
    void op(spv::Op op, int ops_size) {
        uint32_t lower = op & 0xFFFFu;
        uint32_t upper = (ops_size << 16) & 0xFFFF0000u;
        output_word(lower | upper);
    }

    void ref_id(SpvId id) {
        assert(id != 0);
        output_word(id);
    }

    void literal_name(std::string_view str) {
        int i = 0;
        uint32_t cword = 0;
        for (char c : str) {
            cword = cword | (c & 0xFF) << (i * 8);
            i++;
            if (i == 4) {
                output_word(cword);
                cword = 0;
                i = 0;
            }
        }
        output_word(cword);
    }

    void literal_int(uint32_t i) {
        output_word(i);
    }
};



struct SpvFileBuilder {

    enum UniqueDeclTag {
        NONE,
        FN_TYPE,
        PTR_TYPE,
        DEF_ARR_TYPE,
        CONSTANT,
        CONSTANT_COMPOSITE,
    };

    /// Prevents duplicate declarations
    struct UniqueDeclKey {
        UniqueDeclTag tag;
        std::vector<uint32_t> members;

        bool operator==(const UniqueDeclKey &b) const {
            return tag == b.tag && members == b.members;
        }
    };

    struct UniqueDeclKeyHasher {
        size_t operator() (const UniqueDeclKey& key) const {
            size_t acc = 0;
            for (auto id : key.members)
                acc ^= std::hash<uint32_t>{}(id);
            return std::hash<size_t>{}(key.tag) ^ acc;
        }
    };

    SpvFileBuilder() {}
    SpvFileBuilder(const SpvFileBuilder&) = delete;

    SpvId generate_fresh_id() { return { bound++ }; }

    void name(SpvId id, std::string_view str) {
        assert(id < bound);
        debug_names.op(spv::Op::OpName, 2 + div_roundup(str.size() + 1, 4));
        debug_names.ref_id(id);
        debug_names.literal_name(str);
    }

    SpvId declare_bool_type() {
        types_constants.op(spv::Op::OpTypeBool, 2);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        return id;
    }

    SpvId declare_int_type(int width, bool signed_) {
        types_constants.op(spv::Op::OpTypeInt, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(width);
        types_constants.literal_int(signed_ ? 1 : 0);
        return id;
    }

    SpvId declare_float_type(int width) {
        types_constants.op(spv::Op::OpTypeFloat, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(width);
        return id;
    }

    SpvId declare_ptr_type(spv::StorageClass storage_class, SpvId element_type) {
        auto key = UniqueDeclKey { PTR_TYPE, { element_type, (uint32_t) storage_class } };
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.op(spv::Op::OpTypePointer, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(storage_class);
        types_constants.ref_id(element_type);
        unique_decls[key] = id;
        return id;
    }

    SpvId declare_array_type(SpvId element_type, SpvId dim) {
        auto key = UniqueDeclKey { DEF_ARR_TYPE, { element_type, dim } };
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.op(spv::Op::OpTypeArray, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(element_type);
        types_constants.ref_id(dim);
        unique_decls[key] = id;
        return id;
    }

    SpvId declare_fn_type(std::vector<SpvId> dom, SpvId codom) {
        auto key = UniqueDeclKey { FN_TYPE, {} };
        for (auto d : dom) key.members.push_back(d);
        key.members.push_back(codom);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;

        types_constants.op(spv::Op::OpTypeFunction, 3 + dom.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(codom);
        for (auto arg : dom)
            types_constants.ref_id(arg);
        unique_decls[key] = id;
        return id;
    }

    SpvId declare_struct_type(std::vector<SpvId> elements) {
        types_constants.op(spv::Op::OpTypeStruct, 2 + elements.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        for (auto arg : elements)
            types_constants.ref_id(arg);
        return id;
    }

    SpvId declare_vector_type(SpvId component_type, uint32_t dim) {
        types_constants.op(spv::Op::OpTypeVector, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(component_type);
        types_constants.literal_int(dim);
        return id;
    }

    void decorate(SpvId target, spv::Decoration decoration, std::vector<uint32_t> extra = {}) {
        annotations.op(spv::Op::OpDecorate, 3 + extra.size());
        annotations.ref_id(target);
        annotations.literal_int(decoration);
        for (auto e : extra)
            annotations.literal_int(e);
    }

    void decorate_member(SpvId target, uint32_t member, spv::Decoration decoration, std::vector<uint32_t> extra = {}) {
        annotations.op(spv::Op::OpMemberDecorate, 4 + extra.size());
        annotations.ref_id(target);
        annotations.literal_int(member);
        annotations.literal_int(decoration);
        for (auto e : extra)
            annotations.literal_int(e);
    }

    SpvId debug_string(std::string string) {
        debug_string_source.op(spv::Op::OpString, 2 + div_roundup(string.size() + 1, 4));
        auto id = generate_fresh_id();
        debug_string_source.ref_id(id);
        debug_string_source.literal_name(string);
        return id;
    }

    SpvId bool_constant(SpvId type, bool value) {
        types_constants.op(value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        return id;
    }

    SpvId constant(SpvId type, std::vector<uint32_t> bit_pattern) {
        auto key = UniqueDeclKey { CONSTANT, bit_pattern };
        key.members.push_back(type);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.op(spv::Op::OpConstant, 3 + bit_pattern.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        for (auto arg : bit_pattern)
            types_constants.literal_int(arg);
        unique_decls[key] = id;
        return id;
    }

    SpvId constant_composite(SpvId type, std::vector<SpvId> ops) {
        auto key = UniqueDeclKey { CONSTANT_COMPOSITE, {} };
        key.members.push_back(type);
        for (auto op : ops) key.members.push_back(op);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.op(spv::Op::OpConstantComposite, 3 + ops.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        for (auto op : ops)
            types_constants.ref_id(op);
        unique_decls[key] = id;
        return id;
    }

    SpvId variable(SpvId type, spv::StorageClass storage_class) {
        types_constants.op(spv::Op::OpVariable, 4);
        types_constants.ref_id(type);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(storage_class);
        return id;
    }

    SpvId declare_void_type() {
        types_constants.op(spv::Op::OpTypeVoid, 2);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        return id;
    }

    SpvId define_function(SpvFnBuilder& fn_builder);

    void declare_entry_point(spv::ExecutionModel execution_model, SpvId entry_point, std::string name, std::vector<SpvId> interface) {
        entry_points.op(spv::Op::OpEntryPoint, 3 + div_roundup(name.size() + 1, 4) + interface.size());
        entry_points.literal_int(execution_model);
        entry_points.ref_id(entry_point);
        entry_points.literal_name(name);
        for (auto i : interface)
            entry_points.ref_id(i);
    }

    void execution_mode(SpvId entry_point, spv::ExecutionMode execution_mode, std::vector<uint32_t> payloads) {
        entry_points.op(spv::Op::OpExecutionMode, 3 + payloads.size());
        entry_points.ref_id(entry_point);
        entry_points.literal_int(execution_mode);
        for (auto d : payloads)
            entry_points.literal_int(d);
    }

    void capability(spv::Capability cap) {
        auto found = capabilities_set.find(cap);
        if (found != capabilities_set.end())
            return;
        capabilities.op(spv::Op::OpCapability, 2);
        capabilities.data_.push_back(cap);
        capabilities_set.insert(cap);
    }

    void extension(std::string name) {
        auto found = extensions_set.find(name);
        if (found != extensions_set.end())
            return;
        extensions.op(spv::Op::OpExtension, 1 + div_roundup(name.size() + 1, 4));
        extensions.literal_name(name);
        extensions_set.insert(name);
    }

    uint32_t version = spv::Version;

    spv::AddressingModel addressing_model = spv::AddressingModel::AddressingModelLogical;
    spv::MemoryModel memory_model = spv::MemoryModel::MemoryModelSimple;

protected:
    SpvId extended_import(std::string name) {
        auto found = extended_instruction_sets.find(name);
        if (found != extended_instruction_sets.end())
            return found->second;
        ext_inst_import.op(spv::Op::OpExtInstImport, 2 + div_roundup(name.size() + 1, 4));
        auto id = generate_fresh_id();
        ext_inst_import.ref_id(id);
        ext_inst_import.literal_name(name);
        extended_instruction_sets[name] = id;
        return id;
    }

private:
    std::ostream* output_ = nullptr;
    uint32_t bound = 1;

    // Ordered as per https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.pdf#subsection.2.4
    SpvSectionBuilder capabilities;
    SpvSectionBuilder extensions;
    SpvSectionBuilder ext_inst_import;
    SpvSectionBuilder entry_points;
    SpvSectionBuilder execution_modes;
    SpvSectionBuilder debug_string_source;
    SpvSectionBuilder debug_names;
    SpvSectionBuilder debug_module_processed;
    SpvSectionBuilder annotations;
    SpvSectionBuilder types_constants;
    SpvSectionBuilder fn_decls;
    SpvSectionBuilder fn_defs;

    // SPIR-V disallows duplicate non-aggregate type declarations, we protect against these with this
    std::unordered_map<UniqueDeclKey, SpvId, UniqueDeclKeyHasher> unique_decls;
    std::unordered_map<std::string, SpvId> extended_instruction_sets;
    std::unordered_set<spv::Capability> capabilities_set;
    std::unordered_set<std::string> extensions_set;

    void output_word_le(uint32_t word) {
        output_->put((word >> 0) & 0xFFu);
        output_->put((word >> 8) & 0xFFu);
        output_->put((word >> 16) & 0xFFu);
        output_->put((word >> 24) & 0xFFu);
    }

    void output_section(SpvSectionBuilder& section) {
        for (auto& word : section.data_) {
            output_word_le(word);
        }
    }
public:
    void finish(std::ostream& output) {
        output_ = &output;
        SpvSectionBuilder memory_model_section;
        memory_model_section.op(spv::Op::OpMemoryModel, 3);
        memory_model_section.data_.push_back(addressing_model);
        memory_model_section.data_.push_back(memory_model);

        output_word_le(spv::MagicNumber);
        output_word_le(version); // TODO: target a specific spirv version
        output_word_le(uint32_t(0)); // TODO get a magic number ?
        output_word_le(bound);
        output_word_le(uint32_t(0)); // instruction schema padding

        output_section(capabilities);
        output_section(extensions);
        output_section(ext_inst_import);
        output_section(memory_model_section);
        output_section(entry_points);
        output_section(execution_modes);
        output_section(debug_string_source);
        output_section(debug_names);
        output_section(debug_module_processed);
        output_section(annotations);
        output_section(types_constants);
        output_section(fn_decls);
        output_section(fn_defs);
    }

    friend SpvBasicBlockBuilder;
};

struct SpvBasicBlockBuilder : public SpvSectionBuilder {
    explicit SpvBasicBlockBuilder(SpvFileBuilder& file_builder)
            : file_builder(file_builder)
    {}

    SpvFileBuilder& file_builder;

    struct Phi {
        SpvId type;
        SpvId value;
        std::vector<std::pair<SpvId, SpvId>> preds;
    };
    std::vector<Phi*> phis;
    SpvId label;

    SpvId undef(SpvId type) {
        op(spv::Op::OpUndef, 3);
        ref_id(type);
        auto id = generate_fresh_id();
        ref_id(id);
        return id;
    }

    SpvId composite(SpvId aggregate_t, std::vector<SpvId>& elements) {
        op(spv::Op::OpCompositeConstruct, 3 + elements.size());
        ref_id(aggregate_t);
        auto id = generate_fresh_id();
        ref_id(id);
        for (auto e : elements)
            ref_id(e);
        return id;
    }

    SpvId extract(SpvId target_type, SpvId composite, std::vector<uint32_t> indices) {
        op(spv::Op::OpCompositeExtract, 4 + indices.size());
        ref_id(target_type);
        auto id = generate_fresh_id();
        ref_id(id);
        ref_id(composite);
        for (auto i : indices)
            literal_int(i);
        return id;
    }

    SpvId insert(SpvId target_type, SpvId object, SpvId composite, std::vector<uint32_t> indices) {
        op(spv::Op::OpCompositeInsert, 5 + indices.size());
        ref_id(target_type);
        auto id = generate_fresh_id();
        ref_id(id);
        ref_id(object);
        ref_id(composite);
        for (auto i : indices)
            literal_int(i);
        return id;
    }

    SpvId vector_extract_dynamic(SpvId target_type, SpvId vector, SpvId index) {
        op(spv::Op::OpVectorExtractDynamic, 5);
        ref_id(target_type);
        auto id = generate_fresh_id();
        ref_id(id);
        ref_id(vector);
        ref_id(index);
        return id;
    }

    SpvId vector_insert_dynamic(SpvId target_type, SpvId vector, SpvId component, SpvId index) {
        op(spv::Op::OpVectorInsertDynamic, 6);
        ref_id(target_type);
        auto id = generate_fresh_id();
        ref_id(id);
        ref_id(vector);
        ref_id(component);
        ref_id(index);
        return id;
    }

    // Used for almost all conversion operations
    SpvId convert(spv::Op op_, SpvId target_type, SpvId value) {
        op(op_, 4);
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(value);
        return id;
    }

    SpvId access_chain(SpvId target_type, SpvId element, std::vector<SpvId> indexes) {
        op(spv::Op::OpAccessChain, 4 + indexes.size());
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(element);
        for (auto index : indexes)
            ref_id(index);
        return id;
    }

    SpvId ptr_access_chain(SpvId target_type, SpvId base, SpvId element, std::vector<SpvId> indexes) {
        op(spv::Op::OpPtrAccessChain, 5 + indexes.size());
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(base);
        ref_id(element);
        for (auto index : indexes)
            ref_id(index);
        return id;
    }

    SpvId load(SpvId target_type, SpvId pointer, std::vector<uint32_t> operands = {}) {
        op(spv::Op::OpLoad, 4 + operands.size());
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(pointer);
        for (auto op : operands)
            literal_int(op);
        return id;
    }

    void store(SpvId value, SpvId pointer, std::vector<uint32_t> operands = {}) {
        op(spv::Op::OpStore, 3 + operands.size());
        ref_id(pointer);
        ref_id(value);
        for (auto op : operands)
            literal_int(op);
    }

    SpvId binop(spv::Op op_, SpvId result_type, SpvId lhs, SpvId rhs) {
        op(op_, 5);
        auto id = generate_fresh_id();
        ref_id(result_type);
        ref_id(id);
        ref_id(lhs);
        ref_id(rhs);
        return id;
    }

    void branch(SpvId target) {
        op(spv::Op::OpBranch, 2);
        ref_id(target);
    }

    void branch_conditional(SpvId condition, SpvId true_target, SpvId false_target) {
        op(spv::Op::OpBranchConditional, 4);
        ref_id(condition);
        ref_id(true_target);
        ref_id(false_target);
    }

    void branch_switch(SpvId selector, SpvId default_case, std::vector<uint32_t> literals, std::vector<uint32_t> cases) {
        assert(literals.size() == cases.size());
        op(spv::Op::OpSwitch, 3 + literals.size() * 2);
        ref_id(selector);
        ref_id(default_case);
        for (size_t i = 0; i < literals.size(); i++) {
            ref_id(literals[i]);
            ref_id(cases[i]);
        }
    }

    void selection_merge(SpvId merge_bb, spv::SelectionControlMask selection_control) {
        op(spv::Op::OpSelectionMerge, 3);
        ref_id(merge_bb);
        literal_int(selection_control);
    }

    void loop_merge(SpvId merge_bb, SpvId continue_bb, spv::LoopControlMask loop_control, std::vector<uint32_t> loop_control_ops) {
        op(spv::Op::OpLoopMerge, 4 + loop_control_ops.size());
        ref_id(merge_bb);
        ref_id(continue_bb);
        literal_int(loop_control);

        for (auto e : loop_control_ops)
            literal_int(e);
    }

    SpvId call(SpvId return_type, SpvId callee, std::vector<SpvId> arguments) {
        op(spv::Op::OpFunctionCall, 4 + arguments.size());
        auto id = generate_fresh_id();
        ref_id(return_type);
        ref_id(id);
        ref_id(callee);

        for (auto a : arguments)
            ref_id(a);
        return id;
    }

    SpvId ext_instruction(SpvId return_type, ExtendedInstruction instr, std::vector<SpvId> arguments);

    void return_void() {
        op(spv::Op::OpReturn, 1);
    }

    void return_value(SpvId value) {
        op(spv::Op::OpReturnValue, 2);
        ref_id(value);
    }

    void unreachable() {
        op(spv::Op::OpUnreachable, 1);
    }

private:
    SpvId generate_fresh_id();

protected:
    SpvId ext_instruction(SpvId return_type, SpvId set, uint32_t instruction, std::vector<SpvId> arguments) {
        op(spv::Op::OpExtInst, 5 + arguments.size());
        auto id = generate_fresh_id();
        ref_id(return_type);
        ref_id(id);
        ref_id(set);
        literal_int(instruction);
        for (auto a : arguments)
            ref_id(a);
        return id;
    }
};

struct SpvFnBuilder {
    explicit SpvFnBuilder(SpvFileBuilder* file_builder)
            : file_builder(file_builder)
    {
        function_id = generate_fresh_id();
    }

    SpvFileBuilder* file_builder;
    SpvId function_id;

    SpvId fn_type;
    SpvId fn_ret_type;
    std::vector<SpvBasicBlockBuilder*> bbs_to_emit;

    // Contains OpFunctionParams
    SpvSectionBuilder header;

    SpvSectionBuilder variables;

    SpvId parameter(SpvId param_type) {
        header.op(spv::Op::OpFunctionParameter, 3);
        auto id = generate_fresh_id();
        header.ref_id(param_type);
        header.ref_id(id);
        return id;
    }

    SpvId variable(SpvId type, spv::StorageClass storage_class) {
        variables.op(spv::Op::OpVariable, 4);
        variables.ref_id(type);
        auto id = generate_fresh_id();
        variables.ref_id(id);
        variables.literal_int(storage_class);
        return id;
    }

private:
    SpvId generate_fresh_id();
};

inline SpvId SpvFileBuilder::define_function(SpvFnBuilder &fn_builder) {
    fn_defs.op(spv::Op::OpFunction, 5);
    fn_defs.ref_id(fn_builder.fn_ret_type);
    fn_defs.ref_id(fn_builder.function_id);
    fn_defs.data_.push_back(spv::FunctionControlMaskNone);
    fn_defs.ref_id(fn_builder.fn_type);

    // Includes stuff like OpFunctionParameters
    for (auto w : fn_builder.header.data_)
        fn_defs.data_.push_back(w);

    bool first = true;
    for (auto& bb : fn_builder.bbs_to_emit) {
        fn_defs.op(spv::Op::OpLabel, 2);
        fn_defs.ref_id(bb->label);

        if (first) {
            for (auto w : fn_builder.variables.data_)
                fn_defs.data_.push_back(w);
            first = false;
        }

        for (auto& phi : bb->phis) {
            fn_defs.op(spv::Op::OpPhi, 3 + 2 * phi->preds.size());
            fn_defs.ref_id(phi->type);
            fn_defs.ref_id(phi->value);
            assert(!phi->preds.empty());
            for (auto& [pred_value, pred_label] : phi->preds) {
                fn_defs.ref_id(pred_value);
                fn_defs.ref_id(pred_label);
            }
        }

        for (auto w : bb->data_)
            fn_defs.data_.push_back(w);
    }

    fn_defs.op(spv::Op::OpFunctionEnd, 1);
    return fn_builder.function_id;
}


inline SpvId SpvBasicBlockBuilder::generate_fresh_id() {
    return file_builder.generate_fresh_id();
}

inline SpvId SpvFnBuilder::generate_fresh_id() {
    return file_builder->generate_fresh_id();
}

inline SpvId SpvBasicBlockBuilder::ext_instruction(SpvId return_type, ExtendedInstruction instr, std::vector<SpvId> arguments) {
    return ext_instruction(return_type, file_builder.extended_import(instr.set_name), instr.id, arguments);
}

}
