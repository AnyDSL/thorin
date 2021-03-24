#include <spirv/unified1/spirv.hpp>

#include <string>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <ostream>
#include <memory>

namespace thorin::spirv::builder {

struct SpvId { uint32_t id; };

struct SpvSectionBuilder;
struct SpvBasicBlockBuilder;
struct SpvFnBuilder;
struct SpvFileBuilder;

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
        assert(id.id != 0);
        output_word(id.id);
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

    SpvId variable(SpvId type, spv::StorageClass storage_class) {
        op(spv::Op::OpVariable, 4);
        ref_id(type);
        auto id = generate_fresh_id();
        ref_id(id);
        literal_int(storage_class);
        return id;
    }

    SpvId bitcast(SpvId target_type, SpvId value) {
        op(spv::Op::OpBitcast, 4);
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(value);
        return id;
    }

    SpvId load(SpvId target_type, SpvId pointer) {
        op(spv::Op::OpLoad, 4);
        auto id = generate_fresh_id();
        ref_id(target_type);
        ref_id(id);
        ref_id(pointer);
        return id;
    }

    void store(SpvId value, SpvId pointer) {
        op(spv::Op::OpStore, 3);
        auto id = generate_fresh_id();
        ref_id(pointer);
        ref_id(value);
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

    void return_void() {
        op(spv::Op::OpReturn, 1);
    }

    void return_value(SpvId value) {
        op(spv::Op::OpReturnValue, 2);
        ref_id(value);
    }

private:
    SpvId generate_fresh_id();
};

struct SpvFnBuilder {
public:
    SpvId fn_type;
    SpvId fn_ret_type;
    std::vector<SpvBasicBlockBuilder*> bbs_to_emit;

    // Contains OpFunctionParams
    SpvSectionBuilder header;
};

struct SpvFileBuilder {
    SpvFileBuilder()
    : void_type(declare_void_type())
    {}
    SpvFileBuilder(const SpvFileBuilder&) = delete;

    SpvId generate_fresh_id() { return { bound++ }; }

    void name(SpvId id, std::string_view str) {
        assert(id.id < bound);
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
        types_constants.op(spv::Op::OpTypePointer, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.literal_int(storage_class);
        types_constants.ref_id(element_type);
        return id;
    }

    SpvId declare_array_type(SpvId element_type, SpvId dim) {
        types_constants.op(spv::Op::OpTypeArray, 4);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(element_type);
        types_constants.ref_id(dim);
        return id;
    }

    SpvId declare_fn_type(std::vector<SpvId>& dom, SpvId codom) {
        types_constants.op(spv::Op::OpTypeFunction, 3 + dom.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        types_constants.ref_id(codom);
        for (auto arg : dom)
            types_constants.ref_id(arg);
        return id;
    }

    SpvId declare_struct_type(std::vector<SpvId>& elements) {
        types_constants.op(spv::Op::OpTypeStruct, 2 + elements.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        for (auto arg : elements)
            types_constants.ref_id(arg);
        return id;
    }

    SpvId bool_constant(SpvId type, bool value) {
        types_constants.op(value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        return id;
    }

    SpvId constant(SpvId type, std::vector<uint32_t>&& bit_pattern) {
        types_constants.op(spv::Op::OpConstant, 3 + bit_pattern.size());
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        for (auto arg : bit_pattern)
            types_constants.data_.push_back(arg);
        return id;
    }

    SpvId define_function(SpvFnBuilder& fn_builder) {
        fn_defs.op(spv::Op::OpFunction, 5);
        fn_defs.ref_id(fn_builder.fn_ret_type);
        auto id = generate_fresh_id();
        fn_defs.ref_id(id);
        fn_defs.data_.push_back(spv::FunctionControlMaskNone);
        fn_defs.ref_id(fn_builder.fn_type);

        // Includes stuff like OpFunctionParameters
        for (auto w : fn_builder.header.data_)
            fn_defs.data_.push_back(w);

        for (auto& bb : fn_builder.bbs_to_emit) {
            fn_defs.op(spv::Op::OpLabel, 2);
            fn_defs.ref_id(bb->label);

            for (auto& phi : bb->phis) {
                fn_defs.op(spv::Op::OpPhi, 3 + 2 * phi->preds.size());
                fn_defs.ref_id(phi->type);
                fn_defs.ref_id(phi->value);
                printf("Phi %d\n", phi->value);
                assert(phi->preds.size() > 0);
                for (auto& [pred_value, pred_label] : phi->preds) {
                    fn_defs.ref_id(pred_value);
                    fn_defs.ref_id(pred_label);
                }
            }

            for (auto w : bb->data_)
                fn_defs.data_.push_back(w);
        }

        fn_defs.op(spv::Op::OpFunctionEnd, 1);
        return id;
    }

    void capability(spv::Capability cap) {
        capabilities.op(spv::Op::OpCapability, 2);
        capabilities.data_.push_back(cap);
    }

    spv::AddressingModel addressing_model = spv::AddressingModel::AddressingModelLogical;
    spv::MemoryModel memory_model = spv::MemoryModel::MemoryModelSimple;

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

    SpvId declare_void_type() {
        types_constants.op(spv::Op::OpTypeVoid, 2);
        auto id = generate_fresh_id();
        types_constants.ref_id(id);
        return id;
    }

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
    const SpvId void_type;

    void finish(std::ostream& output) {
        output_ = &output;
        SpvSectionBuilder memory_model_section;
        memory_model_section.op(spv::Op::OpMemoryModel, 3);
        memory_model_section.data_.push_back(addressing_model);
        memory_model_section.data_.push_back(memory_model);

        output_word_le(spv::MagicNumber);
        output_word_le(spv::Version); // TODO: target a specific spirv version
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
};

inline SpvId SpvBasicBlockBuilder::generate_fresh_id() {
    return file_builder.generate_fresh_id();
}

}