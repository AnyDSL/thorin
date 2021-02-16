#include "thorin/be/spirv/spirv.h"
#include "thorin/analyses/scope.h"

#include <spirv/unified1/spirv.hpp>

#include <iostream>

int div_roundup(int a, int b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

namespace thorin::spirv {

struct SpvId { uint32_t id; };

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
};

struct SpvFileBuilder {
    SpvFileBuilder(std::ostream& output) : output_(output) {}

    SpvId fresh_id() { return { bound++ }; }

    void name(SpvId id, std::string_view str) {
        assert(id.id < bound);
        debug_names.op(spv::Op::OpName, 2 + div_roundup(str.size() + 1, 4));
        debug_names.ref_id(id);
        debug_names.literal_name(str);
    }

    SpvId declare_bool_type() {
        types_constants.op(spv::Op::OpTypeBool, 2);
        auto id = fresh_id();
        types_constants.ref_id(id);
        return id;
    }

    void capability(spv::Capability cap) {
        capabilities.op(spv::Op::OpCapability, 2);
        capabilities.data_.push_back(cap);
    }

    spv::AddressingModel addressing_model = spv::AddressingModel::AddressingModelLogical;
    spv::MemoryModel memory_model = spv::MemoryModel::MemoryModelSimple;

private:
    std::ostream& output_;
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
    std::vector<SpvSectionBuilder> fn_decls;
    std::vector<SpvSectionBuilder> fn_defs;

    void output_word_le(uint32_t word) {
        output_.put((word >> 0) & 0xFFu);
        output_.put((word >> 8) & 0xFFu);
        output_.put((word >> 16) & 0xFFu);
        output_.put((word >> 24) & 0xFFu);
    }

    void output_section(SpvSectionBuilder& section) {
        for (auto& word : section.data_) {
            output_word_le(word);
        }
    }
public:
    void finish() {
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
        for (auto& decl : fn_decls)
            output_section(decl);
        for (auto& def : fn_defs)
            output_section(def);
    }
};

CodeGen::CodeGen(thorin::World& world, Cont2Config& kernel_config, bool debug)
    : thorin::CodeGen(world, debug)
{}

void CodeGen::emit(std::ostream& out) {
    auto builder = SpvFileBuilder(out);

    builder.capability(spv::Capability::CapabilityShader);
    builder.capability(spv::Capability::CapabilityLinkage);

    builder.name(builder.declare_bool_type(), "test");

    Scope::for_each(world(), [&](const Scope& scope) { emit(scope); });

    builder.finish();
}

void CodeGen::emit(const thorin::Scope& scope) {

}

}
