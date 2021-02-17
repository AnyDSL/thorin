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
    if (auto llvm_type = types_.lookup(type)) return *llvm_type;

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
            //llvm_type = llvm::PointerType::get(convert(ptr->pointee()), convert_addr_space(ptr->addr_space()));
            break;
        }
        case Node_IndefiniteArrayType: {
            assert(false && "TODO");
            //llvm_type = llvm::ArrayType::get(convert(type->as<ArrayType>()->elem_type()), 0);
            //return types_[type] = llvm_type;
        }
        case Node_DefiniteArrayType: {
            assert(false && "TODO");
            auto array = type->as<DefiniteArrayType>();
            //llvm_type = llvm::ArrayType::get(convert(array->elem_type()), array->dim());
            //return types_[type] = llvm_type;
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
                    else                            assert(false && "Didn't we refactor this out yet by making functions single-argument ?"); //ret = llvm::StructType::get(context(), ret_types);
                } else
                    ops.push_back(convert(op));
            }
            assert(ret);

            if (type->tag() == Node_FnType) {
                return types_[type] = builder_->declare_fn_type(ops, *ret);
            }

            assert(false && "TODO: handle closure mess");
            /* auto env_type = convert(Closure::environment_type(world()));
            ops.push_back(env_type);
            auto fn_type = llvm::FunctionType::get(ret, ops, false);
            auto ptr_type = llvm::PointerType::get(fn_type, 0);
            llvm_type = llvm::StructType::get(context(), { ptr_type, env_type });
            return types_[type] = llvm_type;*/
        }

        case Node_StructType: {
            assert(false && "TODO");
            /*auto struct_type = type->as<StructType>();
            auto llvm_struct = llvm::StructType::create(context());

            // important: memoize before recursing into element types to avoid endless recursion
            assert(!types_.contains(struct_type) && "type already converted");
            types_[struct_type] = llvm_struct;

            Array<llvm::Type*> llvm_types(struct_type->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(struct_type->op(i));
            llvm_struct->setBody(llvm_ref(llvm_types));
            return llvm_struct;*/
        }

        case Node_TupleType: {
            assert(false && "TODO");
            /*auto tuple = type->as<TupleType>();
            Array<llvm::Type*> llvm_types(tuple->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(tuple->op(i));
            llvm_type = llvm::StructType::get(context(), llvm_ref(llvm_types));
            return types_[tuple] = llvm_type;*/
        }

        case Node_VariantType: {
            assert(false && "TODO");
            /*assert(type->num_ops() > 0);

            // Max alignment/size constraints respectively in the variant type alternatives dictate the ones to use for the overall type
            size_t max_align = 0, max_size = 0;

            auto layout = module().getDataLayout();
            llvm::Type* max_align_type;
            for (auto op : type->ops()) {
                auto op_type = convert(op);
                size_t size  = layout.getTypeAllocSize(op_type);
                size_t align = layout.getABITypeAlignment(op_type);
                // Favor types that are not empty
                if (align > max_align || (align == max_align && max_align_type->isEmptyTy())) {
                    max_align_type = op_type;
                    max_align = align;
                }
                max_size = std::max(max_size, size);
            }

            auto rem_size = max_size - layout.getTypeAllocSize(max_align_type);
            auto union_type = rem_size > 0
                              ? llvm::StructType::get(context(), llvm::ArrayRef<llvm::Type*> { max_align_type, llvm::ArrayType::get(llvm::Type::getInt8Ty(context()), rem_size)})
                              : llvm::StructType::get(context(), llvm::ArrayRef<llvm::Type*> { max_align_type });

            auto tag_type = type->num_ops() < (1_u64 <<  8) ? llvm::Type::getInt8Ty (context()) :
                            type->num_ops() < (1_u64 << 16) ? llvm::Type::getInt16Ty(context()) :
                            type->num_ops() < (1_u64 << 32) ? llvm::Type::getInt32Ty(context()) :
                            llvm::Type::getInt64Ty(context());

            return llvm::StructType::get(context(), { union_type, tag_type });*/
        }

        default:
            THORIN_UNREACHABLE;
    }

    return types_[type] = spv_type;
}

void CodeGen::emit(const thorin::Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());
    //auto fct = llvm::cast<llvm::Function>(emit(entry_));
    auto fn = SpvFnBuilder { };
    fn.fn_type = convert(entry_->type());
    fn.fn_ret_type = builder_->void_type; // TODO !!!

    current_fn_ = &fn;

    //cont2llvm_.clear();
    auto conts = schedule(scope);

    // map all bb-like continuations to llvm bb stubs and handle params/phis
    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;

        auto [i, b] = fn.bbs.emplace(cont, std::make_unique<SpvBasicBlockBuilder>(*builder_));
        SpvBasicBlockBuilder& bb = *i->second;
        SpvId label = bb.label();
        builder_->name(label, cont->name().c_str());
        fn.labels.emplace(cont, label);
        //auto bb = llvm::BasicBlock::Create(context(), cont->name().c_str(), fct);
        //auto [i, succ] = cont2llvm_.emplace(cont, std::pair(bb, std::make_unique<llvm::IRBuilder<>>(context())));
        assert(b);
        // auto& irbuilder = *i->second.second;
        // irbuilder.SetInsertPoint(bb);

        //if (debug())
        //    irbuilder.SetCurrentDebugLocation(llvm::DebugLoc::get(cont->loc().begin.row, cont->loc().begin.row, discope));

        /*if (entry_ == cont) {
            auto arg = fct->arg_begin();
            for (auto param : entry_->params()) {
                if (is_mem(param) || is_unit(param)) {
                    def2llvm_[param] = nullptr;
                } else if (param->order() == 0) {
                    auto argv = &*arg;
                    auto value = map_param(fct, argv, param);
                    if (value == argv) {
                        arg->setName(param->unique_name()); // use param
                        def2llvm_[param] = &*arg++;
                    } else {
                        def2llvm_[param] = value;           // use provided value
                    }
                }
            }
        } else {
            for (auto param : cont->params()) {
                if (is_mem(param) || is_unit(param)) {
                    def2llvm_[param] = nullptr;
                } else {
                    // do not bother reserving anything (the 0 below) - it's a tiny optimization nobody cares about
                    auto phi = irbuilder.CreatePHI(convert(param->type()), 0, param->name().c_str());
                    def2llvm_[param] = phi;
                }
            }
        }*/
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

void CodeGen::emit_epilogue(Continuation* continuation, SpvBasicBlockBuilder& bb) {
    /*auto&& bb_ib = cont2llvm_[continuation];
    auto bb = bb_ib->first;
    auto& irbuilder = *bb_ib->second;*/

    if (continuation->callee() == entry_->ret_param()) { // return
        std::vector<SpvId> values;
        std::vector<SpvId> types;

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
