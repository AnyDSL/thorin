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

    struct Phi {
        SpvId type;
        SpvId value;
        std::vector<std::pair<SpvId, SpvId>> preds;
    };
    GIDMap<const Param*, Phi> phis;
    DefMap<SpvId> args;

    SpvId label() {
        op(spv::Op::OpLabel, 2);
        auto id = generate_fresh_id();
        ref_id(id);
        return id;
    }

    SpvId composite(SpvId aggregate_t, std::vector<SpvId>& elements) {
        op(spv::Op::OpLabel, 3 + elements.size());
        ref_id(aggregate_t);
        auto id = generate_fresh_id();
        ref_id(id);
        for (auto e : elements)
            ref_id(e);
        return id;
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
    SpvId fn_type;
    SpvId fn_ret_type;
    ContinuationMap<std::unique_ptr<SpvBasicBlockBuilder>> bbs;
    ContinuationMap<SpvId> labels;
    DefMap<SpvId> params;

    // Contains OpFunctionParams
    SpvSectionBuilder header;
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

    SpvId bool_constant(SpvId type, bool value) {
        types_constants.op(value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, 3);
        auto id = generate_fresh_id();
        types_constants.ref_id(type);
        types_constants.ref_id(id);
        return id;
    }

    SpvId constant(SpvId type, std::vector<uint32_t>&& bit_pattern) {
        types_constants.op(spv::Op::OpConstant, 4 + bit_pattern.size());
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

        for (auto& [cont, bb] : fn_builder.bbs) {
            for (auto [param, phi] : bb->phis) {
                fn_defs.op(spv::Op::OpPhi, 3 + 2 * phi.preds.size());
                fn_defs.ref_id(phi.type);
                fn_defs.ref_id(phi.value);
                for (auto& [pred_value, pred_label] : phi.preds) {
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

SpvId SpvBasicBlockBuilder::generate_fresh_id() {
    return file_builder.generate_fresh_id();
}

CodeGen::CodeGen(thorin::World& world, Cont2Config&, bool debug)
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
            break;
        }

        case Node_StructType: {
            std::vector<SpvId> types;
            for (auto elem : type->as<StructType>()->ops())
                types.push_back(convert(elem));
            spv_type = builder_->declare_struct_type(types);
            // TODO debug info
            break;
        }

        case Node_TupleType: {
            std::vector<SpvId> types;
            for (auto elem : type->as<TupleType>()->ops())
                types.push_back(convert(elem));
            spv_type = builder_->declare_struct_type(types);
            // TODO debug info
            break;
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

        if (entry_ == cont) {
            for (auto param : entry_->params()) {
                if (is_mem(param) || is_unit(param)) {
                    // Nothing
                } else if (param->order() == 0) {
                    auto param_t = convert(param->type());
                    fn.header.op(spv::Op::OpFunctionParameter, 3);
                    auto id = builder_->generate_fresh_id();
                    fn.header.ref_id(param_t);
                    fn.header.ref_id(id);
                    fn.params[param] = id;
                }
            }
        } else {
            for (auto param : cont->params()) {
                if (is_mem(param) || is_unit(param)) {
                    // Nothing
                } else {
                    // OpPhi requires the full list of predecessors (values, labels)
                    // We don't have that yet! But we will need the Phi node identifier to build the basic blocks ...
                    // To solve this we generate an id for the phi node now, but defer emission of it to a later stage
                    bb.phis[param] = { convert(param->type()), builder_->generate_fresh_id(), {} };
                }
            }
        }
    }

    Scheduler new_scheduler(scope);
    swap(scheduler_, new_scheduler);

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;
        assert(cont == entry_ || cont->is_basicblock());
        emit_epilogue(cont, *fn.bbs[cont]->get());
    }

    // Wire up Phi nodes
    for (auto& [cont, bb]: fn.bbs) {
        for (auto [param, phi] : bb->phis) {
            assert(param->order() == 0);
            for (auto pred : cont->preds()) {
                auto& pred_bb = *fn.bbs[pred];
                auto arg = pred->arg(param->index());
                phi.preds.emplace_back(*pred_bb->args[arg], *fn.labels[pred]);
            }
        }
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
    if (continuation->callee() == entry_->ret_param()) {
        std::vector<SpvId> values;

        for (auto arg : continuation->args()) {
            assert(arg->order() == 0);
            if (is_mem(arg) || is_unit(arg))
                continue;
            auto val = emit(arg, bb);
            values.emplace_back(val);
        }

        switch (values.size()) {
            case 0:  bb.return_void();      break;
            case 1:  bb.return_value(values[0]); break;
            default: bb.return_value(bb.composite(current_fn_->fn_ret_type, values));
        }
    }
    else if (continuation->callee() == world().branch()) {
        auto cond = emit(continuation->arg(0), bb);
        bb.args[continuation->arg(0)] = cond;
        auto tbb = *current_fn_->labels[continuation->arg(1)->as_continuation()];
        auto fbb = *current_fn_->labels[continuation->arg(2)->as_continuation()];
        bb.branch_conditional(cond, tbb, fbb);
    } /*else if (continuation->callee()->isa<Continuation>() &&
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
    } */
    else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        for (auto& arg : continuation->args()) {
            if (is_mem(arg) || is_unit(arg)) continue;
            bb.args[arg] = emit(arg, bb);
        }
        bb.branch(*current_fn_->labels[callee]);
    } /*else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
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
    else {
        assert(false && "epilogue not implemented for this");
    }
}

SpvId CodeGen::emit(const Def* def, SpvBasicBlockBuilder& bb) {
    if (auto bin = def->isa<BinOp>()) {
        SpvId lhs = emit(bin->lhs(), bb);
        SpvId rhs = emit(bin->rhs(), bb);
        SpvId result_type = convert(def->type());

        if (auto cmp = bin->isa<Cmp>()) {
            auto type = cmp->lhs()->type();
            if (is_type_s(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return bb.binop(spv::Op::OpIEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb.binop(spv::Op::OpINotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb.binop(spv::Op::OpSGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb.binop(spv::Op::OpSGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb.binop(spv::Op::OpSLessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb.binop(spv::Op::OpSLessThanEqual    , result_type, lhs, rhs);
                }
            } else if (is_type_u(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return bb.binop(spv::Op::OpIEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb.binop(spv::Op::OpINotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb.binop(spv::Op::OpUGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb.binop(spv::Op::OpUGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb.binop(spv::Op::OpULessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb.binop(spv::Op::OpULessThanEqual    , result_type, lhs, rhs);
                }
            } else if (is_type_f(type)) {
                switch (cmp->cmp_tag()) {
                    // TODO look into the NaN story
                    case Cmp_eq: return bb.binop(spv::Op::OpFOrdEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb.binop(spv::Op::OpFOrdNotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb.binop(spv::Op::OpFOrdGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb.binop(spv::Op::OpFOrdGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb.binop(spv::Op::OpFOrdLessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb.binop(spv::Op::OpFOrdLessThanEqual    , result_type, lhs, rhs);
                }
            } else if (type->isa<PtrType>()) {
                assertf(false, "Physical pointers are unsupported");
            } else if(is_type_bool(type)) {
                switch (cmp->cmp_tag()) {
                    // TODO look into the NaN story
                    case Cmp_eq: return bb.binop(spv::Op::OpLogicalEqual    , result_type, lhs, rhs);
                    case Cmp_ne: return bb.binop(spv::Op::OpLogicalNotEqual , result_type, lhs, rhs);
                    default: THORIN_UNREACHABLE;
                }
                assertf(false, "TODO: should we emulate the other comparison ops ?");
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();

            if (is_type_f(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb.binop(spv::Op::OpFAdd, result_type, lhs, rhs);
                    case ArithOp_sub: return bb.binop(spv::Op::OpFSub, result_type, lhs, rhs);
                    case ArithOp_mul: return bb.binop(spv::Op::OpFMul, result_type, lhs, rhs);
                    case ArithOp_div: return bb.binop(spv::Op::OpFDiv, result_type, lhs, rhs);
                    case ArithOp_rem: return bb.binop(spv::Op::OpFRem, result_type, lhs, rhs);
                    case ArithOp_and:
                    case ArithOp_or:
                    case ArithOp_xor:
                    case ArithOp_shl:
                    case ArithOp_shr: THORIN_UNREACHABLE;
                }
            }

            if (is_type_s(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb.binop(spv::Op::OpIAdd                 , result_type, lhs, rhs);
                    case ArithOp_sub: return bb.binop(spv::Op::OpISub                 , result_type, lhs, rhs);
                    case ArithOp_mul: return bb.binop(spv::Op::OpIMul                 , result_type, lhs, rhs);
                    case ArithOp_div: return bb.binop(spv::Op::OpSDiv                 , result_type, lhs, rhs);
                    case ArithOp_rem: return bb.binop(spv::Op::OpSRem                 , result_type, lhs, rhs);
                    case ArithOp_and: return bb.binop(spv::Op::OpBitwiseAnd           , result_type, lhs, rhs);
                    case ArithOp_or:  return bb.binop(spv::Op::OpBitwiseOr            , result_type, lhs, rhs);
                    case ArithOp_xor: return bb.binop(spv::Op::OpBitwiseXor           , result_type, lhs, rhs);
                    case ArithOp_shl: return bb.binop(spv::Op::OpShiftLeftLogical     , result_type, lhs, rhs);
                    case ArithOp_shr: return bb.binop(spv::Op::OpShiftRightArithmetic , result_type, lhs, rhs);
                }
            } else if (is_type_u(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb.binop(spv::Op::OpIAdd              , result_type, lhs, rhs);
                    case ArithOp_sub: return bb.binop(spv::Op::OpISub              , result_type, lhs, rhs);
                    case ArithOp_mul: return bb.binop(spv::Op::OpIMul              , result_type, lhs, rhs);
                    case ArithOp_div: return bb.binop(spv::Op::OpUDiv              , result_type, lhs, rhs);
                    case ArithOp_rem: return bb.binop(spv::Op::OpUMod              , result_type, lhs, rhs);
                    case ArithOp_and: return bb.binop(spv::Op::OpBitwiseAnd        , result_type, lhs, rhs);
                    case ArithOp_or:  return bb.binop(spv::Op::OpBitwiseOr         , result_type, lhs, rhs);
                    case ArithOp_xor: return bb.binop(spv::Op::OpBitwiseXor        , result_type, lhs, rhs);
                    case ArithOp_shl: return bb.binop(spv::Op::OpShiftLeftLogical  , result_type, lhs, rhs);
                    case ArithOp_shr: return bb.binop(spv::Op::OpShiftRightLogical , result_type, lhs, rhs);
                }
            } else if(is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_and: return bb.binop(spv::Op::OpLogicalAnd      , result_type, lhs, rhs);
                    case ArithOp_or:  return bb.binop(spv::Op::OpLogicalOr       , result_type, lhs, rhs);
                    // Note: there is no OpLogicalXor
                    case ArithOp_xor: return bb.binop(spv::Op::OpLogicalNotEqual , result_type, lhs, rhs);
                    default: THORIN_UNREACHABLE;
                }
            }
            THORIN_UNREACHABLE;
        }
    }
    if (auto primlit = def->isa<PrimLit>()) {
        Box box = primlit->value();
        auto type = convert(def->type());
        SpvId constant;
        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     constant = bb.file_builder.bool_constant(type, box.get_bool()); break;
            case PrimType_ps8:  case PrimType_qs8:  assertf(false, "not implemented yet");
            case PrimType_pu8:  case PrimType_qu8:  assertf(false, "not implemented yet");
            case PrimType_ps16: case PrimType_qs16: assertf(false, "not implemented yet");
            case PrimType_pu16: case PrimType_qu16: assertf(false, "not implemented yet");
            case PrimType_ps32: case PrimType_qs32: constant = bb.file_builder.constant(type, { static_cast<unsigned int>(box.get_s32()) }); break;
            case PrimType_pu32: case PrimType_qu32: constant = bb.file_builder.constant(type, { static_cast<unsigned int>(box.get_u32()) }); break;
            case PrimType_ps64: case PrimType_qs64: assertf(false, "not implemented yet");
            case PrimType_pu64: case PrimType_qu64: assertf(false, "not implemented yet");
            case PrimType_pf16: case PrimType_qf16: assertf(false, "not implemented yet");
            case PrimType_pf32: case PrimType_qf32: assertf(false, "not implemented yet");
            case PrimType_pf64: case PrimType_qf64: assertf(false, "not implemented yet");
        }
        return constant;
    }
    assertf(false, "Incomplete emit(def) definition");
}

}
