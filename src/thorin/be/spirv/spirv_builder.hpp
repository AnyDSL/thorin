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
using Id = uint32_t;

struct SectionBuilder;
struct BasicBlockBuilder;
struct FnBuilder;
struct FileBuilder;

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

struct SectionBuilder {
    std::vector<uint32_t> data_;

private:
    void output_word(uint32_t word) {
        data_.push_back(word);
    }
public:
    void begin_op(spv::Op op, int size_in_words) {
        uint32_t lower = op & 0xFFFFu;
        uint32_t upper = (size_in_words << 16) & 0xFFFF0000u;
        output_word(lower | upper);
    }

    void ref_id(Id id) {
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

struct FileBuilder {
    enum UniqueDeclTag {
        NONE,
        VOID_TYPE,
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

    FileBuilder() {}
    FileBuilder(const FileBuilder&) = delete;

    Id generate_fresh_id() { return { bound++ }; }

    void name(Id id, std::string_view str) {
        assert(id < bound);
        debug_names.begin_op(spv::Op::OpName, 2 + div_roundup(str.size() + 1, 4));
        debug_names.ref_id(id);
        debug_names.literal_name(str);
    }

    Id declare_bool_type() {
        types_constants.begin_op(spv::Op::OpTypeBool, 2);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        return id;
    }

    Id declare_int_type(int width, bool signed_) {
        types_constants.begin_op(spv::Op::OpTypeInt, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(width);
        types_constants.literal_int(signed_ ? 1 : 0);
        return id;
    }

    Id declare_float_type(int width) {
        types_constants.begin_op(spv::Op::OpTypeFloat, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(width);
        return id;
    }

    Id declare_ptr_type(spv::StorageClass storage_class, Id element_type) {
        auto key = UniqueDeclKey { PTR_TYPE, { element_type, (uint32_t) storage_class } };
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.begin_op(spv::Op::OpTypePointer, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(storage_class);
        types_constants.ref_id(element_type);
        unique_decls[key] = id;
        return id;
    }

    Id declare_array_type(Id element_type, Id dim) {
        auto key = UniqueDeclKey { DEF_ARR_TYPE, { element_type, dim } };
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.begin_op(spv::Op::OpTypeArray, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(element_type);
        types_constants.ref_id(dim);
        unique_decls[key] = id;
        return id;
    }

    Id declare_fn_type(std::vector<Id> dom, Id codom) {
        auto key = UniqueDeclKey { FN_TYPE, {} };
        for (auto d : dom) key.members.push_back(d);
        key.members.push_back(codom);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;

        types_constants.begin_op(spv::Op::OpTypeFunction, 3 + dom.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(codom);
        for (auto arg : dom)
            types_constants.ref_id(arg);
        unique_decls[key] = id;
        return id;
    }

    Id declare_struct_type(std::vector<Id> elements) {
        types_constants.begin_op(spv::Op::OpTypeStruct, 2 + elements.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        for (auto arg : elements)
            types_constants.ref_id(arg);
        return id;
    }

    Id declare_vector_type(Id component_type, uint32_t dim) {
        types_constants.begin_op(spv::Op::OpTypeVector, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(component_type);
        types_constants.literal_int(dim);
        return id;
    }

    void decorate(Id target, spv::Decoration decoration, std::vector<uint32_t> extra = {}) {
        annotations.begin_op(spv::Op::OpDecorate, 3 + extra.size());
        annotations.ref_id(target);
        annotations.literal_int(decoration);
        for (auto e : extra)
            annotations.literal_int(e);
    }

    void decorate_member(Id target, uint32_t member, spv::Decoration decoration, std::vector<uint32_t> extra = {}) {
        annotations.begin_op(spv::Op::OpMemberDecorate, 4 + extra.size());
        annotations.ref_id(target);
        annotations.literal_int(member);
        annotations.literal_int(decoration);
        for (auto e : extra)
            annotations.literal_int(e);
    }

    Id debug_string(std::string string) {
        debug_string_source.begin_op(spv::Op::OpString, 2 + div_roundup(string.size() + 1, 4));
        auto id = generate_fresh_id();
        debug_string_source.ref_id(id);
        debug_string_source.literal_name(string);
        return id;
    }

    Id bool_constant(Id type, bool value) {
        types_constants.begin_op(value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        return id;
    }

    Id constant(Id type, std::vector<uint32_t> bit_pattern) {
        auto key = UniqueDeclKey { CONSTANT, bit_pattern };
        key.members.push_back(type);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.begin_op(spv::Op::OpConstant, 3 + bit_pattern.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        for (auto arg : bit_pattern)
            types_constants.literal_int(arg);
        unique_decls[key] = id;
        return id;
    }

    Id constant_composite(Id type, std::vector<Id> ops) {
        auto key = UniqueDeclKey { CONSTANT_COMPOSITE, {} };
        key.members.push_back(type);
        for (auto op : ops) key.members.push_back(op);
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.begin_op(spv::Op::OpConstantComposite, 3 + ops.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        for (auto op : ops)
            types_constants.ref_id(op);
        unique_decls[key] = id;
        return id;
    }

    Id variable(Id type, spv::StorageClass storage_class) {
        types_constants.begin_op(spv::Op::OpVariable, 4);
        types_constants.ref_id(type);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(storage_class);
        return id;
    }

    Id declare_void_type() {
        auto key = UniqueDeclKey { VOID_TYPE, { } };
        if (auto iter = unique_decls.find(key); iter != unique_decls.end()) return iter->second;
        types_constants.begin_op(spv::Op::OpTypeVoid, 2);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        unique_decls[key] = id;
        return id;
    }

    Id define_function(FnBuilder& fn_builder);

    void declare_entry_point(spv::ExecutionModel execution_model, Id entry_point, std::string name, std::vector<Id> interface) {
        entry_points.begin_op(spv::Op::OpEntryPoint, 3 + div_roundup(name.size() + 1, 4) + interface.size());
        entry_points.literal_int(execution_model);
        entry_points.ref_id(entry_point);
        entry_points.literal_name(name);
        for (auto i : interface)
            entry_points.ref_id(i);
    }

    void execution_mode(Id entry_point, spv::ExecutionMode execution_mode, std::vector<uint32_t> payloads) {
        entry_points.begin_op(spv::Op::OpExecutionMode, 3 + payloads.size());
        entry_points.ref_id(entry_point);
        entry_points.literal_int(execution_mode);
        for (auto d : payloads)
            entry_points.literal_int(d);
    }

    void capability(spv::Capability cap) {
        auto found = capabilities_set.find(cap);
        if (found != capabilities_set.end())
            return;
        capabilities.begin_op(spv::Op::OpCapability, 2);
        capabilities.data_.push_back(cap);
        capabilities_set.insert(cap);
    }

    void extension(std::string name) {
        auto found = extensions_set.find(name);
        if (found != extensions_set.end())
            return;
        extensions.begin_op(spv::Op::OpExtension, 1 + div_roundup(name.size() + 1, 4));
        extensions.literal_name(name);
        extensions_set.insert(name);
    }

    uint32_t version = spv::Version;

    spv::AddressingModel addressing_model = spv::AddressingModel::AddressingModelLogical;
    spv::MemoryModel memory_model = spv::MemoryModel::MemoryModelSimple;

protected:
    Id extended_import(std::string name) {
        auto found = extended_instruction_sets.find(name);
        if (found != extended_instruction_sets.end())
            return found->second;
        ext_inst_import.begin_op(spv::Op::OpExtInstImport, 2 + div_roundup(name.size() + 1, 4));
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
    SectionBuilder capabilities;
    SectionBuilder extensions;
    SectionBuilder ext_inst_import;
    SectionBuilder entry_points;
    SectionBuilder execution_modes;
    SectionBuilder debug_string_source;
    SectionBuilder debug_names;
    SectionBuilder debug_module_processed;
    SectionBuilder annotations;
    SectionBuilder types_constants;
    SectionBuilder fn_decls;
    SectionBuilder fn_defs;

    // SPIR-V disallows duplicate non-aggregate type declarations, we protect against these with this
    std::unordered_map<UniqueDeclKey, Id, UniqueDeclKeyHasher> unique_decls;
    std::unordered_map<std::string, Id> extended_instruction_sets;
    std::unordered_set<spv::Capability> capabilities_set;
    std::unordered_set<std::string> extensions_set;

    void output_word_le(uint32_t word) {
        output_->put((word >> 0) & 0xFFu);
        output_->put((word >> 8) & 0xFFu);
        output_->put((word >> 16) & 0xFFu);
        output_->put((word >> 24) & 0xFFu);
    }

    void output_section(SectionBuilder& section) {
        for (auto& word : section.data_) {
            output_word_le(word);
        }
    }
public:
    void finish(std::ostream& output) {
        output_ = &output;
        SectionBuilder memory_model_section;
        memory_model_section.begin_op(spv::Op::OpMemoryModel, 3);
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

    friend BasicBlockBuilder;
};

struct BasicBlockBuilder : public SectionBuilder {
    explicit BasicBlockBuilder(FileBuilder& file_builder)
            : file_builder(file_builder), terminator(*this) {}

    FileBuilder& file_builder;

    struct Phi {
        Id type;
        Id value;
        std::vector<std::pair<Id, Id>> preds;
    };
    std::vector<Phi*> phis;
    Id label;

    Id undef(Id type) {
        begin_op(spv::Op::OpUndef, 3);
        ref_id(type);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        return id;
    }

    Id composite(Id aggregate_t, std::vector<Id>& elements) {
        begin_op(spv::Op::OpCompositeConstruct, 3 + elements.size());
        ref_id(aggregate_t);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        for (auto e : elements)
            ref_id(e);
        return id;
    }

    Id extract(Id target_type, Id composite, std::vector<uint32_t> indices) {
        begin_op(spv::Op::OpCompositeExtract, 4 + indices.size());
        ref_id(target_type);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        ref_id(composite);
        for (auto i : indices)
            literal_int(i);
        return id;
    }

    Id insert(Id target_type, Id object, Id composite, std::vector<uint32_t> indices) {
        begin_op(spv::Op::OpCompositeInsert, 5 + indices.size());
        ref_id(target_type);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        ref_id(object);
        ref_id(composite);
        for (auto i : indices)
            literal_int(i);
        return id;
    }

    Id vector_extract_dynamic(Id target_type, Id vector, Id index) {
        begin_op(spv::Op::OpVectorExtractDynamic, 5);
        ref_id(target_type);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        ref_id(vector);
        ref_id(index);
        return id;
    }

    Id vector_insert_dynamic(Id target_type, Id vector, Id component, Id index) {
        begin_op(spv::Op::OpVectorInsertDynamic, 6);
        ref_id(target_type);
        auto id = file_builder.generate_fresh_id();
        ref_id(id);
        ref_id(vector);
        ref_id(component);
        ref_id(index);
        return id;
    }

    // Used for almost all conversion operations
    Id convert(spv::Op op_, Id target_type, Id value) {
        begin_op(op_, 4);
        auto id = file_builder.generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(value);
        return id;
    }

    Id access_chain(Id target_type, Id element, std::vector<Id> indexes) {
        begin_op(spv::Op::OpAccessChain, 4 + indexes.size());
        auto id = file_builder.generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(element);
        for (auto index : indexes)
            ref_id(index);
        return id;
    }

    Id ptr_access_chain(Id target_type, Id base, Id element, std::vector<Id> indexes) {
        begin_op(spv::Op::OpPtrAccessChain, 5 + indexes.size());
        auto id = file_builder.generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(base);
        ref_id(element);
        for (auto index : indexes)
            ref_id(index);
        return id;
    }

    Id load(Id target_type, Id pointer, std::vector<uint32_t> operands = {}) {
        begin_op(spv::Op::OpLoad, 4 + operands.size());
        auto id = file_builder.generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(pointer);
        for (auto op : operands)
            literal_int(op);
        return id;
    }

    void store(Id value, Id pointer, std::vector<uint32_t> operands = {}) {
        begin_op(spv::Op::OpStore, 3 + operands.size());
        ref_id(pointer);
        ref_id(value);
        for (auto op : operands)
            literal_int(op);
    }

    Id binop(spv::Op op_, Id result_type, Id lhs, Id rhs) {
        begin_op(op_, 5);
        auto id = file_builder.generate_fresh_id();
        ref_id(result_type);
        ref_id(id);
        ref_id(lhs);
        ref_id(rhs);
        return id;
    }

    Id call(Id return_type, Id callee, std::vector<Id> arguments) {
        begin_op(spv::Op::OpFunctionCall, 4 + arguments.size());
        auto id = file_builder.generate_fresh_id();
        ref_id(return_type);
        ref_id(id);
        ref_id(callee);

        for (auto a : arguments)
            ref_id(a);
        return id;
    }

    Id ext_instruction(Id return_type, ExtendedInstruction instr, std::vector<Id> arguments);

    struct TerminatorBuilder : public SectionBuilder {
        TerminatorBuilder(BasicBlockBuilder& bb) : bb(bb) {}

        void branch(Id target) {
            begin_op(spv::Op::OpBranch, 2);
            ref_id(target);
        }

        void branch_conditional(Id condition, Id true_target, Id false_target) {
            begin_op(spv::Op::OpBranchConditional, 4);
            ref_id(condition);
            ref_id(true_target);
            ref_id(false_target);
        }

        void branch_switch(Id selector, Id default_case, std::vector<uint32_t> literals, std::vector<uint32_t> cases) {
            assert(literals.size() == cases.size());
            begin_op(spv::Op::OpSwitch, 3 + literals.size() * 2);
            ref_id(selector);
            ref_id(default_case);
            for (size_t i = 0; i < literals.size(); i++) {
                ref_id(literals[i]);
                ref_id(cases[i]);
            }
        }

        void selection_merge(Id merge_bb, spv::SelectionControlMask selection_control) {
            begin_op(spv::Op::OpSelectionMerge, 3);
            ref_id(merge_bb);
            literal_int(selection_control);
        }

        void loop_merge(Id merge_bb, Id continue_bb, spv::LoopControlMask loop_control, std::vector<uint32_t> loop_control_ops) {
            begin_op(spv::Op::OpLoopMerge, 4 + loop_control_ops.size());
            ref_id(merge_bb);
            ref_id(continue_bb);
            literal_int(loop_control);

            for (auto e : loop_control_ops)
                literal_int(e);
        }

        void return_void() {
            begin_op(spv::Op::OpReturn, 1);
        }

        void return_value(Id value) {
            begin_op(spv::Op::OpReturnValue, 2);
            ref_id(value);
        }

        void unreachable() {
            begin_op(spv::Op::OpUnreachable, 1);
        }

    private:
        BasicBlockBuilder& bb;
    };

    TerminatorBuilder terminator;

protected:
    Id ext_instruction(Id return_type, Id set, uint32_t instruction, std::vector<Id> arguments) {
        begin_op(spv::Op::OpExtInst, 5 + arguments.size());
        auto id = file_builder.generate_fresh_id();
        ref_id(return_type);
        ref_id(id);
        ref_id(set);
        literal_int(instruction);
        for (auto a : arguments)
            ref_id(a);
        return id;
    }
};

struct FnBuilder {
    explicit FnBuilder(FileBuilder& file_builder)
            : file_builder(file_builder)
    {
        function_id = file_builder.generate_fresh_id();
    }

    FileBuilder& file_builder;
    Id function_id;

    Id fn_type;
    Id fn_ret_type;
    std::vector<BasicBlockBuilder*> bbs_to_emit;

    // Contains OpFunctionParams
    SectionBuilder header;

    SectionBuilder variables;

    Id parameter(Id param_type) {
        header.begin_op(spv::Op::OpFunctionParameter, 3);
        auto id = file_builder.generate_fresh_id();
        header.ref_id(param_type);
        header.ref_id(id);
        return id;
    }

    Id variable(Id type, spv::StorageClass storage_class) {
        variables.begin_op(spv::Op::OpVariable, 4);
        variables.ref_id(type);
        auto id = file_builder.generate_fresh_id();
        variables.ref_id(id);
        variables.literal_int(storage_class);
        return id;
    }
};

inline Id FileBuilder::define_function(FnBuilder &fn_builder) {
    fn_defs.begin_op(spv::Op::OpFunction, 5);
    fn_defs.ref_id(fn_builder.fn_ret_type);
    fn_defs.ref_id(fn_builder.function_id);
    fn_defs.data_.push_back(spv::FunctionControlMaskNone);
    fn_defs.ref_id(fn_builder.fn_type);

    // Includes stuff like OpFunctionParameters
    for (auto w : fn_builder.header.data_)
        fn_defs.data_.push_back(w);

    bool first = true;
    for (auto& bb : fn_builder.bbs_to_emit) {
        fn_defs.begin_op(spv::Op::OpLabel, 2);
        fn_defs.ref_id(bb->label);

        if (first) {
            for (auto w : fn_builder.variables.data_)
                fn_defs.data_.push_back(w);
            first = false;
        }

        for (auto& phi : bb->phis) {
            fn_defs.begin_op(spv::Op::OpPhi, 3 + 2 * phi->preds.size());
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

        for (auto w : bb->terminator.data_)
            fn_defs.data_.push_back(w);
    }

    fn_defs.begin_op(spv::Op::OpFunctionEnd, 1);
    return fn_builder.function_id;
}

inline Id BasicBlockBuilder::ext_instruction(Id return_type, ExtendedInstruction instr, std::vector<Id> arguments) {
    return ext_instruction(return_type, file_builder.extended_import(instr.set_name), instr.id, arguments);
}

}
