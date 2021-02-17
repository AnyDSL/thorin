#include "thorin/be/spirv/spirv.h"
#include "thorin/analyses/scope.h"

#include <spirv/unified1/spirv.hpp>

#include <iostream>
#include <thorin/analyses/schedule.h>

int div_roundup(int a, int b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

namespace thorin::spirv {

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

    SpvId label() {
        op(spv::Op::OpLabel, 2);
        auto id = generate_fresh_id();
        ref_id(id);
        return id;
    }

    void return_void() {
        op(spv::Op::OpReturn, 1);
    }
private:
    SpvId generate_fresh_id();
};

struct SpvFnBuilder {
    SpvId fn_type;
    SpvId fn_ret_type;
    ContinuationMap<std::unique_ptr<SpvBasicBlockBuilder>> bbs;
    ContinuationMap<SpvId> labels;
};

struct SpvFileBuilder {
    SpvFileBuilder()
        : void_type(declare_void_type())
    {}

    SpvId generate_fresh_id() { return {bound++ }; }

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

    SpvId define_function(SpvFnBuilder& fn_builder) {
        fn_defs.op(spv::Op::OpFunction, 5);
        fn_defs.ref_id(fn_builder.fn_ret_type);
        auto id = generate_fresh_id();
        fn_defs.ref_id(id);
        fn_defs.data_.push_back(spv::FunctionControlMaskNone);
        fn_defs.ref_id(fn_builder.fn_type);

        // TODO OpFunctionParameters
        for (auto& [cont, bb] : fn_builder.bbs) {
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

SpvId SpvBasicBlockBuilder::generate_fresh_id() {
    return file_builder.generate_fresh_id();
}

CodeGen::CodeGen(thorin::World& world, Cont2Config& kernel_config, bool debug)
    : thorin::CodeGen(world, debug)
{}

void CodeGen::emit(std::ostream& out) {
    SpvFileBuilder builder;
    builder_ = &builder;
    builder_->capability(spv::Capability::CapabilityShader);
    builder_->capability(spv::Capability::CapabilityLinkage);

    Scope::for_each(world(), [&](const Scope& scope) { emit(scope); });

    builder_->finish(out);
    builder_ = nullptr;
}

SpvId CodeGen::convert(const Type* type) {
    if (auto spv_type = types_.lookup(type)) return *spv_type;

    assert(!type->isa<MemType>());
    SpvId spv_type;
    switch (type->tag()) {
        case PrimType_bool:                                                             spv_type = builder_->declare_bool_type(); break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  assert(false && "TODO: look into capabilities to enable this");
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: assert(false && "TODO: look into capabilities to enable this");
        case PrimType_ps32: case PrimType_qs32:                                         spv_type = builder_->declare_int_type(32, true ); break;
        case PrimType_pu32: case PrimType_qu32:                                         spv_type = builder_->declare_int_type(32, false); break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pf16: case PrimType_qf16:                                         assert(false && "TODO: look into capabilities to enable this");
        case PrimType_pf32: case PrimType_qf32:                                         spv_type = builder_->declare_float_type(32); break;
        case PrimType_pf64: case PrimType_qf64:                                         assert(false && "TODO: look into capabilities to enable this");
        case Node_PtrType: {
            auto ptr = type->as<PtrType>();
            assert(false && "TODO");
            break;
        }
        case Node_IndefiniteArrayType: {
            assert(false && "TODO");
            auto array = type->as<IndefiniteArrayType>();
            //return types_[type] = spv_type;
        }
        case Node_DefiniteArrayType: {
            assert(false && "TODO");
            auto array = type->as<DefiniteArrayType>();
            //return types_[type] = spv_type;
        }

        case Node_ClosureType:
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type->as<FnType>();
            std::unique_ptr<SpvId> ret;
            std::vector<SpvId> ops;
            for (auto op : fn->ops()) {
                if (op->isa<MemType>() || op == world().unit()) continue;
                auto fn = op->isa<FnType>();
                if (fn && !op->isa<ClosureType>()) {
                    assert(!ret && "only one 'return' supported");
                    std::vector<SpvId> ret_types;
                    for (auto fn_op : fn->ops()) {
                        if (fn_op->isa<MemType>() || fn_op == world().unit()) continue;
                        ret_types.push_back(convert(fn_op));
                    }
                    if (ret_types.size() == 0)      ret = std::make_unique<SpvId>(builder_->void_type);
                    else if (ret_types.size() == 1) ret = std::make_unique<SpvId>(ret_types.back());
                    else                            assert(false && "Didn't we refactor this out yet by making functions single-argument ?");
                } else
                    ops.push_back(convert(op));
            }
            assert(ret);

            if (type->tag() == Node_FnType) {
                return types_[type] = builder_->declare_fn_type(ops, *ret);
            }

            assert(false && "TODO: handle closure mess");
        }

        case Node_StructType: {
            assert(false && "TODO");
        }

        case Node_TupleType: {
            assert(false && "TODO");
        }

        case Node_VariantType: {
            assert(false && "TODO");
        }

        default:
            THORIN_UNREACHABLE;
    }

    return types_[type] = spv_type;
}

void CodeGen::emit(const thorin::Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());

    auto fn = SpvFnBuilder { };
    fn.fn_type = convert(entry_->type());
    fn.fn_ret_type = get_codom_type(entry_);

    current_fn_ = &fn;

    auto conts = schedule(scope);

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;

        auto [i, b] = fn.bbs.emplace(cont, std::make_unique<SpvBasicBlockBuilder>(*builder_));
        assert(b);

        SpvBasicBlockBuilder& bb = *i->second;
        SpvId label = bb.label();

        if (debug())
            builder_->name(label, cont->name().c_str());
        fn.labels.emplace(cont, label);

        // TODO prepare phis/params
    }

    Scheduler new_scheduler(scope);
    swap(scheduler_, new_scheduler);

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;
        assert(cont == entry_ || cont->is_basicblock());
        emit_epilogue(cont, *fn.bbs[cont]->get());
    }

    builder_->define_function(fn);
}

SpvId CodeGen::get_codom_type(const Continuation* fn) {
    auto ret_cont_type = fn->ret_param()->type();
    std::vector<SpvId> types;
    for (auto& op : ret_cont_type->ops()) {
        if (op->isa<MemType>() || is_type_unit(op))
            continue;
        assert(op->order() == 0);
        types.push_back(convert(op));
    }
    if (types.empty())
        return builder_->void_type;
    if (types.size() == 1)
        return types[0];
    return builder_->declare_struct_type(types);
}

void CodeGen::emit_epilogue(Continuation* continuation, SpvBasicBlockBuilder& bb) {
    if (continuation->callee() == entry_->ret_param()) { // return
        std::vector<SpvId> types;
        std::vector<SpvId> values;

        for (auto arg : continuation->args()) {
            /*if (auto val = emit_unsafe(arg)) {
                values.emplace_back(val);
                types.emplace_back(val->getType());
            }*/
        }

        switch (values.size()) {
            case 0:  bb.return_void();      break;
                //case 1:  irbuilder.CreateRet(values[0]); break;
            default:
                assert(false && "TODO handle non-void returns");
                /*llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context(), types));

                for (size_t i = 0, e = values.size(); i != e; ++i)
                    agg = irbuilder.CreateInsertValue(agg, values[i], { unsigned(i) });

                irbuilder.CreateRet(agg);*/
        }
    } /*else if (continuation->callee() == world().branch()) {
        auto cond = emit(continuation->arg(0));
        auto tbb = cont2bb(continuation->arg(1)->as_continuation());
        auto fbb = cont2bb(continuation->arg(2)->as_continuation());
        irbuilder.CreateCondBr(cond, tbb, fbb);
    } else if (continuation->callee()->isa<Continuation>() &&
               continuation->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
        auto val = emit(continuation->arg(0));
        auto otherwise_bb = cont2bb(continuation->arg(1)->as_continuation());
        auto match = irbuilder.CreateSwitch(val, otherwise_bb, continuation->num_args() - 2);
        for (size_t i = 2; i < continuation->num_args(); i++) {
            auto arg = continuation->arg(i)->as<Tuple>();
            auto case_const = llvm::cast<llvm::ConstantInt>(emit(arg->op(0)));
            auto case_bb    = cont2bb(arg->op(1)->as_continuation());
            match->addCase(case_const, case_bb);
        }
    } else if (continuation->callee()->isa<Bottom>()) {
        irbuilder.CreateUnreachable();
    } else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        for (size_t i = 0, e = continuation->num_args(); i != e; ++i) {
            if (auto val = emit_unsafe(continuation->arg(i))) emit_phi_arg(irbuilder, callee->param(i), val);
        }
        irbuilder.CreateBr(cont2bb(callee));
    } else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
        auto ret_continuation = emit_intrinsic(irbuilder, continuation);
        irbuilder.CreateBr(cont2bb(ret_continuation));
    } else { // function/closure call
        // put all first-order args into an array
        std::vector<llvm::Value*> args;
        const Def* ret_arg = nullptr;
        for (auto arg : continuation->args()) {
            if (arg->order() == 0) {
                if (auto val = emit_unsafe(arg))
                    args.push_back(val);
            } else {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        llvm::CallInst* call = nullptr;
        if (auto callee = continuation->callee()->isa_continuation()) {
            call = irbuilder.CreateCall(emit(callee), args);
            if (callee->is_exported())
                call->setCallingConv(kernel_calling_convention_);
            else if (callee->cc() == CC::Device)
                call->setCallingConv(device_calling_convention_);
            else
                call->setCallingConv(function_calling_convention_);
        } else {
            // must be a closure
            auto closure = emit(callee);
            args.push_back(irbuilder.CreateExtractValue(closure, 1));
            call = irbuilder.CreateCall(irbuilder.CreateExtractValue(closure, 0), args);
        }

        // must be call + continuation --- call + return has been removed by codegen_prepare
        auto succ = ret_arg->as_continuation();

        size_t n = 0;
        const Param* last_param = nullptr;
        for (auto param : succ->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            last_param = param;
            n++;
        }

        if (n == 0) {
            irbuilder.CreateBr(cont2bb(succ));
        } else if (n == 1) {
            irbuilder.CreateBr(cont2bb(succ));
            emit_phi_arg(irbuilder, last_param, call);
        } else {
            Array<llvm::Value*> extracts(n);
            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                extracts[j] = irbuilder.CreateExtractValue(call, unsigned(j));
                j++;
            }

            irbuilder.CreateBr(cont2bb(succ));

            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                emit_phi_arg(irbuilder, param, extracts[j]);
                j++;
            }
        }
    }*/

    // new insert point is just before the terminator for all other instructions we have to add later on
    // irbuilder.SetInsertPoint(bb->getTerminator());
}

}
