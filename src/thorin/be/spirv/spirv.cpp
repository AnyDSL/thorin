#include "thorin/be/spirv/spirv.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/cleanup_world.h"

#include <iostream>

namespace thorin {
    void dump_dot(thorin::World &world);
}

namespace thorin::spirv {

/// Used as a dummy SSA value for emitting things like mem/unit
/// Should never make it in the binary files !
constexpr SpvId spv_none { 0 };

// SPIR-V has 3 "kinds" of primitives, and the user may declare arbitrary bitwidths, the following helps in translation:
enum class PrimTypeKind {
    Signed, Unsigned, Float
};
inline PrimTypeKind classify_primtype(const PrimType* type) {
    switch (type->tag()) {
#define THORIN_QS_TYPE(T, M) THORIN_PS_TYPE(T, M)
#define THORIN_PS_TYPE(T, M) \
case PrimType_##T: \
    return PrimTypeKind::Signed; \
    break;
#include "thorin/tables/primtypetable.h"
#undef THORIN_QS_TYPE
#undef THORIN_PS_TYPE

#define THORIN_QU_TYPE(T, M) THORIN_PU_TYPE(T, M)
#define THORIN_PU_TYPE(T, M) \
case PrimType_##T: \
    return PrimTypeKind::Unsigned; \
    break;
#include "thorin/tables/primtypetable.h"
#undef THORIN_QU_TYPE
#undef THORIN_PU_TYPE

#define THORIN_QF_TYPE(T, M) THORIN_PF_TYPE(T, M)
#define THORIN_PF_TYPE(T, M) \
case PrimType_##T: \
    return PrimTypeKind::Float; \
    break;
#include "thorin/tables/primtypetable.h"
#undef THORIN_QF_TYPE
#undef THORIN_PF_TYPE
        default: THORIN_UNREACHABLE;
    }
}
inline const PrimType* get_primtype(World& world, PrimTypeKind kind, int bitwidth, int length) {
#define GET_PRIMTYPE_WITH_KIND(kind) \
switch (bitwidth) { \
    case 8:  return world.type_p##kind##8 (length); \
    case 16: return world.type_p##kind##16(length); \
    case 32: return world.type_p##kind##32(length); \
    case 64: return world.type_p##kind##64(length); \
}

#define GET_PRIMTYPE_WITH_KIND_F(kind) \
switch (bitwidth) { \
    case 8: world.ELOG("8-bit floats do not exist"); \
    case 16: return world.type_p##kind##16(length); \
    case 32: return world.type_p##kind##32(length); \
    case 64: return world.type_p##kind##64(length); \
}

    switch (kind) {
        case PrimTypeKind::Signed:   GET_PRIMTYPE_WITH_KIND(s);   THORIN_UNREACHABLE;
        case PrimTypeKind::Unsigned: GET_PRIMTYPE_WITH_KIND(u);   THORIN_UNREACHABLE;
        case PrimTypeKind::Float:    GET_PRIMTYPE_WITH_KIND_F(f); THORIN_UNREACHABLE;
        default: THORIN_UNREACHABLE;
    }

#undef GET_PRIMTYPE_WITH_KIND
#undef GET_PRIMTYPE_WITH_KIND_F
}

BasicBlockBuilder::BasicBlockBuilder(FnBuilder& fn_builder)
        : builder::SpvBasicBlockBuilder(fn_builder.file_builder), fn_builder(fn_builder), file_builder(fn_builder.file_builder) {
    label = file_builder.generate_fresh_id();
}

FnBuilder::FnBuilder(CodeGen* cg, FileBuilder& file_builder) : builder::SpvFnBuilder(&file_builder), cg(cg), file_builder(file_builder) {}

FileBuilder::FileBuilder(CodeGen* cg) : builder::SpvFileBuilder(), cg(cg) {
    capability(spv::Capability::CapabilityShader);
    capability(spv::Capability::CapabilityVariablePointers);
    capability(spv::Capability::CapabilityPhysicalStorageBufferAddresses);
    // capability(spv::Capability::CapabilityInt16);
    capability(spv::Capability::CapabilityInt64);

    addressing_model = spv::AddressingModelPhysicalStorageBuffer64;
    memory_model = spv::MemoryModel::MemoryModelGLSL450;
}

SpvId FileBuilder::u32_t() {
    if (u32_t_.id == 0)
        u32_t_ = cg->convert(cg->world().type_pu32())->type_id;
    return u32_t_;
}

SpvId FileBuilder::u32_constant(uint32_t pattern) {
    return constant(u32_t(), { pattern });
}

Builtins::Builtins(FileBuilder& builder) {
    auto& world = builder.cg->world();
    auto spv_uvec3_t = builder.cg->convert(world.type_pu32(3));
    auto spv_uint_t = builder.cg->convert(world.type_pu32());
    auto spv_uvec3_pt = builder.declare_ptr_type(spv::StorageClassInput, spv_uvec3_t->type_id);
    auto spv_uint_pt = builder.declare_ptr_type(spv::StorageClassInput, spv_uint_t->type_id);

    // workgroup_size = builder.constant(spv_uvec3_pt, spv::StorageClassInput);
    // builder.decorate(workgroup_size, spv::DecorationBuiltIn, { spv::BuiltInWorkgroupSize });
    // builder.name(workgroup_size, "BuiltInWorkgroupSize");

    num_workgroups = builder.variable(spv_uvec3_pt, spv::StorageClassInput);
    builder.decorate(num_workgroups, spv::DecorationBuiltIn, { spv::BuiltInNumWorkgroups });
    builder.name(num_workgroups, "BuiltInNumWorkgroups");

    workgroup_id = builder.variable(spv_uvec3_pt, spv::StorageClassInput);
    builder.decorate(workgroup_id, spv::DecorationBuiltIn, { spv::BuiltInWorkgroupId });
    builder.name(workgroup_id, "BuiltInWorkgroupId");

    local_id = builder.variable(spv_uvec3_pt, spv::StorageClassInput);
    builder.decorate(local_id, spv::DecorationBuiltIn, { spv::BuiltInLocalInvocationId });
    builder.name(local_id, "BuiltInLocalInvocationId");

    global_id = builder.variable(spv_uvec3_pt, spv::StorageClassInput);
    builder.decorate(global_id, spv::DecorationBuiltIn, { spv::BuiltInGlobalInvocationId });
    builder.name(global_id, "BuiltInGlobalInvocationId");

    local_invocation_index = builder.variable(spv_uint_pt, spv::StorageClassInput);
    builder.decorate(local_invocation_index, spv::DecorationBuiltIn, { spv::BuiltInLocalInvocationIndex });
    builder.name(local_invocation_index, "BuiltInLocalInvocationIndex");
}

ImportedInstructions::ImportedInstructions(FileBuilder& builder) {
    builder.extension("SPV_KHR_non_semantic_info");
    shader_printf = builder.extended_import("NonSemantic.DebugPrintf");
}

CodeGen::CodeGen(thorin::World& world, Cont2Config& kernel_config, bool debug)
        : thorin::CodeGen(world, debug), kernel_config_(kernel_config)
{}

void CodeGen::emit_stream(std::ostream& out) {
    builder_ = std::make_unique<FileBuilder>(this);

    builder_->builtins = std::make_unique<Builtins>(*builder_);
    builder_->imported_instrs = std::make_unique<ImportedInstructions>(*builder_);

    structure_loops();
    structure_flow();
    // cleanup_world(world());
    dump_dot(world());

    Scope::for_each(world(), [&](const Scope& scope) { emit(scope); });

    auto push_constant_arr_type = convert(world().definite_array_type(world().type_pu32(), 128))->type_id;
    auto push_constant_struct_type = builder_->declare_struct_type({ push_constant_arr_type });
    auto push_constant_struct_ptr_type = builder_->declare_ptr_type(spv::StorageClassPushConstant, push_constant_struct_type);
    builder_->name(push_constant_struct_type, "ThorinPushConstant");
    builder_->decorate(push_constant_struct_type, spv::DecorationBlock);
    builder_->decorate_member(push_constant_struct_type, 0, spv::DecorationOffset, { 0 });
    builder_->decorate(push_constant_arr_type, spv::DecorationArrayStride, { 4 });
    auto push_constant_struct_ptr = builder_->variable(push_constant_struct_ptr_type, spv::StorageClassPushConstant);
    builder_->name(push_constant_struct_ptr, "thorin_push_constant_data");

    auto entry_pt_signature = builder_->declare_fn_type({}, builder_->void_type);
    for (auto& cont : world().continuations()) {
        if (cont->is_exported()) {
            assert(defs_.contains(cont) && kernel_config_.contains(cont));
            auto config = kernel_config_.find(cont);

            SpvId callee = defs_[cont];

            FnBuilder fn_builder(this, *builder_.get());
            fn_builder.fn_type = entry_pt_signature;
            fn_builder.fn_ret_type = builder_->void_type;

            BasicBlockBuilder* bb = fn_builder.bbs.emplace_back(std::make_unique<BasicBlockBuilder>(fn_builder)).get();
            fn_builder.bbs_to_emit.push_back(bb);

            // iterate on cont type and extract the arguments
            auto ptr_type = convert(world().ptr_type(world().definite_array_type(world().type_pu32(), 128), 1, 4, AddrSpace::Push))->type_id;
            auto zero = bb->file_builder.u32_constant(0);
            auto arr_ref = bb->access_chain(ptr_type, push_constant_struct_ptr, { zero });
            uint32_t offset = 0;
            std::vector<SpvId> args;
            for (size_t i = 0; i < cont->num_params(); i++) {
                auto param = cont->param(i);
                auto param_type = param->type();
                if (param_type == world().unit() || param_type == world().mem_type() || param_type->isa<FnType>()) continue;
                assert(param_type->order() == 0);
                auto converted = convert(param_type);
                assert(converted->datatype != nullptr);
                SpvId arg = converted->datatype->emit_deserialization(*bb, spv::StorageClassPushConstant, arr_ref, bb->file_builder.u32_constant(offset));
                args.push_back(arg);
                offset += converted->datatype->serialized_size();
            }

            bb->call(builder_->void_type, callee, args);
            bb->return_void();

            builder_->define_function(fn_builder);
            builder_->name(fn_builder.function_id, "entry_point_" + cont->name());

            builder_->declare_entry_point(spv::ExecutionModelGLCompute, fn_builder.function_id, "kernel_main", { push_constant_struct_ptr, builder_->builtins->local_id });

            auto block = config->second->as<GPUKernelConfig>()->block_size();
            std::vector<uint32_t> local_size = {
                (uint32_t) std::get<0>(block),
                (uint32_t) std::get<1>(block),
                (uint32_t) std::get<2>(block),
            };
            builder_->execution_mode(fn_builder.function_id, spv::ExecutionModeLocalSize, local_size);
        }
    }

    builder_->finish(out);
    builder_ = nullptr;
}

void CodeGen::emit(const thorin::Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());

    FnBuilder fn(this, *builder_.get());
    fn.scope = &scope;
    fn.fn_type = convert(entry_->type())->type_id;
    fn.fn_ret_type = get_codom_type(entry_);
    defs_.emplace(scope.entry(), fn.function_id);

    current_fn_ = &fn;

    auto conts = schedule(scope);

    fn.bbs_to_emit.reserve(conts.size());
    fn.bbs.reserve(conts.size());
    auto& bbs = fn.bbs;

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;

        BasicBlockBuilder* bb = bbs.emplace_back(std::make_unique<BasicBlockBuilder>(fn)).get();
        fn.bbs_to_emit.emplace_back(bb);
        auto [i, b] = fn.bbs_map.emplace(cont, bb);
        assert(b);

        if (debug())
            builder_->name(bb->label, cont->name().c_str());
        fn.labels.emplace(cont, bb->label);

        if (entry_ == cont) {
            for (auto param : entry_->params()) {
                if (is_mem(param) || is_unit(param)) {
                    // Nothing
                } else if (param->order() == 0) {
                    auto param_t = convert(param->type());
                    auto id = fn.parameter(param_t->type_id);
                    fn.params[param] = id;
                    if (param->type()->isa<PtrType>()) {
                        builder_->decorate(id, spv::DecorationAliased);
                    }
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
                    auto type = convert(param->type())->type_id;
                    assert(type.id != 0);
                    bb->phis_map[param] = { type, builder_->generate_fresh_id(), {} };
                }
            }
        }
    }

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;
        assert(cont == entry_ || cont->is_basicblock());
        emit_epilogue(cont, fn.bbs_map[cont]);
    }

    for(auto& bb : fn.bbs) {
        for (auto& [param, phi] : bb->phis_map) {
            bb->phis.emplace_back(&phi);
        }
    }
    
    builder_->define_function(fn);
    builder_->name(fn.function_id, scope.entry()->name());
}

SpvId CodeGen::get_codom_type(const Continuation* fn) {
    auto ret_cont_type = fn->ret_param()->type();
    std::vector<SpvId> types;
    for (auto& op : ret_cont_type->ops()) {
        if (op->isa<MemType>() || is_type_unit(op))
            continue;
        assert(op->order() == 0);
        types.push_back(convert(op)->type_id);
    }
    if (types.empty())
        return builder_->void_type;
    if (types.size() == 1)
        return types[0];
    return builder_->declare_struct_type(types);
}

void CodeGen::emit_epilogue(Continuation* continuation, BasicBlockBuilder* bb) {
    // Handles the potential nuances of jumping to another continuation
    auto jump_to_next_cont_with_args = [&](Continuation* succ, std::vector<SpvId> args) {
        bb->branch(current_fn_->labels[succ]);
        for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
            auto param = succ->param(i);
            if (is_mem(param) || is_unit(param))
                continue;
            auto& phi = current_fn_->bbs_map[succ]->phis_map[param];
            phi.preds.emplace_back(args[j], current_fn_->labels[continuation]);
            j++;
        }
    };

    if (continuation->callee() == entry_->ret_param()) {
        std::vector<SpvId> values;

        for (auto arg : continuation->args()) {
            assert(arg->order() == 0);
            auto val = emit(arg, bb);
            if (is_mem(arg) || is_unit(arg))
                continue;
            values.emplace_back(val);
        }

        switch (values.size()) {
            case 0:  bb->return_void();      break;
            case 1:  bb->return_value(values[0]); break;
            default: bb->return_value(bb->composite(current_fn_->fn_ret_type, values));
        }
    } else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        int index = -1;
        for (auto& arg : continuation->args()) {
            index++;
            auto val = emit(arg, bb);
            if (is_mem(arg) || is_unit(arg)) continue;
            bb->args[arg] = val;
            auto* param = callee->param(index);
            auto& phi = current_fn_->bbs_map[callee]->phis_map[param];
            phi.preds.emplace_back(bb->args[arg], current_fn_->labels[continuation]);
        }
        bb->branch(current_fn_->labels[callee]);
    } else if (continuation->callee() == world().branch()) {
        auto& domtree = current_fn_->scope->b_cfg().domtree();
        auto merge_cont = domtree.idom(current_fn_->scope->f_cfg().operator[](continuation))->continuation();
        SpvId merge_bb;
        if (merge_cont == current_fn_->scope->exit()) {
            BasicBlockBuilder* unreachable_merge_bb = current_fn_->bbs.emplace_back(std::make_unique<BasicBlockBuilder>(*current_fn_)).get();
            current_fn_->bbs_to_emit.emplace_back(unreachable_merge_bb);
            builder_->name(unreachable_merge_bb->label, "merge_unreachable" + continuation->name());
            unreachable_merge_bb->unreachable();
            merge_bb = unreachable_merge_bb->label;
        } else {
            // TODO create a dedicated merge bb if this one is the merge blocks for more than 1 selection construct
            merge_bb = current_fn_->labels[merge_cont];
        }

        auto cond = emit(continuation->arg(0), bb);
        bb->args.emplace(continuation->arg(0), cond);
        auto tbb = current_fn_->labels[continuation->arg(1)->as_continuation()];
        auto fbb = current_fn_->labels[continuation->arg(2)->as_continuation()];
        bb->selection_merge(merge_bb,spv::SelectionControlMaskNone);
        bb->branch_conditional(cond, tbb, fbb);
    } else if (continuation->callee()->isa<Continuation>() && continuation->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
        /*auto val = emit(continuation->arg(0));
        auto otherwise_bb = cont2bb(continuation->arg(1)->as_continuation());
        auto match = irbuilder.CreateSwitch(val, otherwise_bb, continuation->num_args() - 2);
        for (size_t i = 2; i < continuation->num_args(); i++) {
            auto arg = continuation->arg(i)->as<Tuple>();
            auto case_const = llvm::cast<llvm::ConstantInt>(emit(arg->op(0)));
            auto case_bb    = cont2bb(arg->op(1)->as_continuation());
            match->addCase(case_const, case_bb);
        }*/
        THORIN_UNREACHABLE;
    } else if (continuation->callee()->isa<Bottom>()) {
        bb->unreachable();
    } else if (continuation->intrinsic() == Intrinsic::SCFLoopHeader) {
        auto merge_label = current_fn_->bbs_map[const_cast<Continuation*>(continuation->attributes_.scf_metadata.loop_header.merge_target)]->label;
        auto continue_label = current_fn_->bbs_map[const_cast<Continuation*>(continuation->attributes_.scf_metadata.loop_header.continue_target)]->label;
        bb->loop_merge(merge_label, continue_label, spv::LoopControlMaskNone, {});

        BasicBlockBuilder* dispatch_bb = current_fn_->bbs.emplace_back(std::make_unique<BasicBlockBuilder>(*current_fn_)).get();

        auto header_bb_location = std::find(current_fn_->bbs_to_emit.begin(), current_fn_->bbs_to_emit.end(), bb);

        current_fn_->bbs_to_emit.emplace(header_bb_location + 1, dispatch_bb);
        builder_->name(dispatch_bb->label, "dispatch_" + continuation->name());
        bb->branch(dispatch_bb->label);
        int targets = continuation->num_ops();
        assert(targets > 0);

        // TODO handle dispatching to multiple targets
        assert(targets == 1);
        auto dispatch_target = continuation->op(0)->as_continuation();
        // Extract the relevant variant & expand the tuple if necessary
        auto arg = world().variant_extract(continuation->param(0), 0);
        auto extracted = emit(arg, dispatch_bb);

        if (dispatch_target->param(0)->type()->equal(arg->type())) {
            auto* param = dispatch_target->param(0);
            auto& phi = current_fn_->bbs_map[dispatch_target]->phis_map[param];
            phi.preds.emplace_back(extracted, dispatch_bb->label);
        } else {
            assert(false && "TODO destructure argument");
        }

        dispatch_bb->branch(current_fn_->bbs_map[dispatch_target]->label);

    } else if (continuation->intrinsic() == Intrinsic::SCFLoopContinue) {
        auto loop_header = continuation->op(0)->as_continuation();
        auto header_label = current_fn_->bbs_map[loop_header]->label;

        auto arg = continuation->param(0);
        bb->args[arg] = emit(arg, bb);
        auto* param = loop_header->param(0);
        auto& phi = current_fn_->bbs_map[loop_header]->phis_map[param];
        phi.preds.emplace_back(bb->args[arg], current_fn_->labels[continuation]);

        bb->branch(header_label);
    } else if (continuation->intrinsic() == Intrinsic::SCFLoopMerge) {

        int targets = continuation->num_ops();
        assert(targets > 0);

        // TODO handle dispatching to multiple targets
        assert(targets == 1);
        auto callee = continuation->op(0)->as_continuation();
        // TODO phis
        bb->branch(current_fn_->bbs_map[callee]->label);
    } else if (auto builtin = continuation->callee()->isa_continuation(); builtin->is_imported()) {
        // Ensure we emit previous memory operations
        assert(is_mem(continuation->arg(0)));
        emit(continuation->arg(0), bb);

        auto productions = emit_builtin(continuation, builtin, bb);
        auto succ = continuation->args().back()->as_continuation();
        jump_to_next_cont_with_args(succ, productions);
    } else if (auto intrinsic = continuation->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
        THORIN_UNREACHABLE;
    } else { // function/closure call
        // put all first-order args into an array
        std::vector<SpvId> call_args;
        const Def* ret_arg = nullptr;
        for (auto arg : continuation->args()) {
            if (arg->order() == 0) {
                auto arg_type = arg->type();
                auto arg_val = emit(arg, bb);
                if (arg_type == world().unit() || arg_type == world().mem_type()) continue;
                call_args.push_back(arg_val);
            } else {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        auto ret_type = get_codom_type(continuation);

        SpvId call_result;
        if (auto called_continuation = continuation->callee()->isa_continuation()) {
            call_result = bb->call(ret_type, emit(called_continuation, bb), call_args);
        } else {
            // must be a closure
            THORIN_UNREACHABLE;

            // auto closure = emit(callee);
            // args.push_back(irbuilder.CreateExtractValue(closure, 1));
            // call = irbuilder.CreateCall(irbuilder.CreateExtractValue(closure, 0), args);
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
            bb->branch(current_fn_->labels[succ]);
        } else if (n == 1) {
            bb->branch(current_fn_->labels[succ]);

            auto& phi = current_fn_->bbs_map[succ]->phis_map[last_param];
            phi.preds.emplace_back(call_result, current_fn_->labels[continuation]);
        } else {
            Array<SpvId> extracts(n);
            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                extracts[j] = bb->extract(convert(param->type())->type_id, call_result, { (uint32_t) j });
                j++;
            }

            bb->branch(current_fn_->labels[succ]);

            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;

                auto& phi = current_fn_->bbs_map[succ]->phis_map[param];
                phi.preds.emplace_back(extracts[j], current_fn_->labels[continuation]);

                j++;
            }
        }
    }
}

SpvId CodeGen::emit(const Def* def, BasicBlockBuilder* bb) {
    if (auto bin = def->isa<BinOp>()) {
        SpvId lhs = emit(bin->lhs(), bb);
        SpvId rhs = emit(bin->rhs(), bb);
        ConvertedType* result_types = convert(def->type());
        SpvId result_type = result_types->type_id;

        if (auto cmp = bin->isa<Cmp>()) {
            auto type = cmp->lhs()->type();
            if (is_type_s(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return bb->binop(spv::Op::OpIEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb->binop(spv::Op::OpINotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb->binop(spv::Op::OpSGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb->binop(spv::Op::OpSGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb->binop(spv::Op::OpSLessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb->binop(spv::Op::OpSLessThanEqual    , result_type, lhs, rhs);
                }
            } else if (is_type_u(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return bb->binop(spv::Op::OpIEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb->binop(spv::Op::OpINotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb->binop(spv::Op::OpUGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb->binop(spv::Op::OpUGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb->binop(spv::Op::OpULessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb->binop(spv::Op::OpULessThanEqual    , result_type, lhs, rhs);
                }
            } else if (is_type_f(type)) {
                switch (cmp->cmp_tag()) {
                    // TODO look into the NaN story
                    case Cmp_eq: return bb->binop(spv::Op::OpFOrdEqual            , result_type, lhs, rhs);
                    case Cmp_ne: return bb->binop(spv::Op::OpFOrdNotEqual         , result_type, lhs, rhs);
                    case Cmp_gt: return bb->binop(spv::Op::OpFOrdGreaterThan      , result_type, lhs, rhs);
                    case Cmp_ge: return bb->binop(spv::Op::OpFOrdGreaterThanEqual , result_type, lhs, rhs);
                    case Cmp_lt: return bb->binop(spv::Op::OpFOrdLessThan         , result_type, lhs, rhs);
                    case Cmp_le: return bb->binop(spv::Op::OpFOrdLessThanEqual    , result_type, lhs, rhs);
                }
            } else if (type->isa<PtrType>()) {
                assertf(false, "Physical pointers are unsupported");
            } else if(is_type_bool(type)) {
                switch (cmp->cmp_tag()) {
                    // TODO look into the NaN story
                    case Cmp_eq: return bb->binop(spv::Op::OpLogicalEqual    , result_type, lhs, rhs);
                    case Cmp_ne: return bb->binop(spv::Op::OpLogicalNotEqual , result_type, lhs, rhs);
                    default: THORIN_UNREACHABLE;
                }
                assertf(false, "TODO: should we emulate the other comparison ops ?");
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();

            if (is_type_f(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb->binop(spv::Op::OpFAdd, result_type, lhs, rhs);
                    case ArithOp_sub: return bb->binop(spv::Op::OpFSub, result_type, lhs, rhs);
                    case ArithOp_mul: return bb->binop(spv::Op::OpFMul, result_type, lhs, rhs);
                    case ArithOp_div: return bb->binop(spv::Op::OpFDiv, result_type, lhs, rhs);
                    case ArithOp_rem: return bb->binop(spv::Op::OpFRem, result_type, lhs, rhs);
                    case ArithOp_and:
                    case ArithOp_or:
                    case ArithOp_xor:
                    case ArithOp_shl:
                    case ArithOp_shr: THORIN_UNREACHABLE;
                }
            }

            if (is_type_s(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb->binop(spv::Op::OpIAdd                 , result_type, lhs, rhs);
                    case ArithOp_sub: return bb->binop(spv::Op::OpISub                 , result_type, lhs, rhs);
                    case ArithOp_mul: return bb->binop(spv::Op::OpIMul                 , result_type, lhs, rhs);
                    case ArithOp_div: return bb->binop(spv::Op::OpSDiv                 , result_type, lhs, rhs);
                    case ArithOp_rem: return bb->binop(spv::Op::OpSRem                 , result_type, lhs, rhs);
                    case ArithOp_and: return bb->binop(spv::Op::OpBitwiseAnd           , result_type, lhs, rhs);
                    case ArithOp_or:  return bb->binop(spv::Op::OpBitwiseOr            , result_type, lhs, rhs);
                    case ArithOp_xor: return bb->binop(spv::Op::OpBitwiseXor           , result_type, lhs, rhs);
                    case ArithOp_shl: return bb->binop(spv::Op::OpShiftLeftLogical     , result_type, lhs, rhs);
                    case ArithOp_shr: return bb->binop(spv::Op::OpShiftRightArithmetic , result_type, lhs, rhs);
                }
            } else if (is_type_u(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return bb->binop(spv::Op::OpIAdd              , result_type, lhs, rhs);
                    case ArithOp_sub: return bb->binop(spv::Op::OpISub              , result_type, lhs, rhs);
                    case ArithOp_mul: return bb->binop(spv::Op::OpIMul              , result_type, lhs, rhs);
                    case ArithOp_div: return bb->binop(spv::Op::OpUDiv              , result_type, lhs, rhs);
                    case ArithOp_rem: return bb->binop(spv::Op::OpUMod              , result_type, lhs, rhs);
                    case ArithOp_and: return bb->binop(spv::Op::OpBitwiseAnd        , result_type, lhs, rhs);
                    case ArithOp_or:  return bb->binop(spv::Op::OpBitwiseOr         , result_type, lhs, rhs);
                    case ArithOp_xor: return bb->binop(spv::Op::OpBitwiseXor        , result_type, lhs, rhs);
                    case ArithOp_shl: return bb->binop(spv::Op::OpShiftLeftLogical  , result_type, lhs, rhs);
                    case ArithOp_shr: return bb->binop(spv::Op::OpShiftRightLogical , result_type, lhs, rhs);
                }
            } else if(is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_and: return bb->binop(spv::Op::OpLogicalAnd      , result_type, lhs, rhs);
                    case ArithOp_or:  return bb->binop(spv::Op::OpLogicalOr       , result_type, lhs, rhs);
                    // Note: there is no OpLogicalXor
                    case ArithOp_xor: return bb->binop(spv::Op::OpLogicalNotEqual , result_type, lhs, rhs);
                    default: THORIN_UNREACHABLE;
                }
            }
            THORIN_UNREACHABLE;
        }
    } else if (auto primlit = def->isa<PrimLit>()) {
        Box box = primlit->value();
        auto type = convert(def->type())->type_id;
        SpvId constant;
        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     constant = bb->file_builder.bool_constant(type, box.get_bool()); break;
            case PrimType_ps8:  case PrimType_qs8:  assertf(false, "not implemented yet");
            case PrimType_pu8:  case PrimType_qu8:  assertf(false, "not implemented yet");
            case PrimType_ps16: case PrimType_qs16: assertf(false, "not implemented yet");
            case PrimType_pu16: case PrimType_qu16: assertf(false, "not implemented yet");
            case PrimType_ps32: case PrimType_qs32: constant = bb->file_builder.constant(type, { static_cast<unsigned int>(box.get_s32()) }); break;
            case PrimType_pu32: case PrimType_qu32: constant = bb->file_builder.constant(type, { static_cast<unsigned int>(box.get_u32()) }); break;
            case PrimType_ps64: case PrimType_qs64:
            case PrimType_pu64: case PrimType_qu64: {
                uint64_t value = static_cast<uint64_t>(box.get_u64());
                uint64_t upper = value >> 32U;
                uint64_t lower = value & 0xFFFFFFFFU;
                constant = bb->file_builder.constant(type, { (uint32_t) lower, (uint32_t) upper });
                break;
            }
            case PrimType_pf16: case PrimType_qf16: assertf(false, "not implemented yet");
            case PrimType_pf32: case PrimType_qf32: assertf(false, "not implemented yet");
            case PrimType_pf64: case PrimType_qf64: assertf(false, "not implemented yet");
        }
        return constant;
    } else if (auto param = def->isa<Param>()) {
        if (is_mem(param)) return spv_none;
        if (auto param_id = current_fn_->params.lookup(param)) {
            assert((*param_id).id != 0);
            return *param_id;
        } else {
            auto val = (*current_fn_->bbs_map[param->continuation()]).phis_map[param].value;
            assert(val.id != 0);
            return val;
        }
    } else if (auto variant = def->isa<Variant>()) {
        auto variant_type = def->type()->as<VariantType>();
        auto variant_datatype = (ProductDatatype*) convert(variant_type)->datatype.get();
        auto tag = builder_->u32_constant(variant->index());

        if (variant_datatype->elements_types.size() > 1) {
            auto alloc_type = convert(world().ptr_type(variant_datatype->elements_types[1]->src_type, 1, 4, AddrSpace::Function))->type_id;
            auto payload_arr = current_fn_->variable(alloc_type, spv::StorageClassFunction);
            auto converted_payload_type = convert(variant_type->op(variant->index()));

            converted_payload_type->datatype->emit_serialization(*bb, spv::StorageClassFunction, payload_arr, bb->file_builder.u32_constant(0), emit(variant->value(), bb));
            auto payload = bb->load(variant_datatype->elements_types[1]->type_id, payload_arr);

            std::vector<SpvId> with_tag = {tag, payload};
            return bb->composite(convert(variant->type())->type_id, with_tag);
        } else {
            // Zero-sized payload case
            std::vector<SpvId> with_tag = { tag };
            return bb->composite(convert(variant->type())->type_id, with_tag);
        }
    } else if (auto vextract = def->isa<VariantExtract>()) {
        auto variant_type = vextract->value()->type()->as<VariantType>();
        auto variant_datatype = (ProductDatatype*) convert(variant_type)->datatype.get();

        auto target_type = convert(def->type());

        assert(variant_datatype->elements_types.size() > 1 && "Can't extract zero-sized datatypes");
        auto alloc_type = convert(world().ptr_type(variant_datatype->elements_types[1]->src_type, 1, 4, AddrSpace::Function))->type_id;
        auto payload_arr = current_fn_->variable(alloc_type, spv::StorageClassFunction);
        auto payload = bb->extract(variant_datatype->elements_types[1]->type_id, emit(vextract->value(), bb), {1});
        bb->store(payload, payload_arr);

        return target_type->datatype->emit_deserialization(*bb, spv::StorageClassFunction, payload_arr, bb->file_builder.u32_constant(0));
    } else if (auto vindex = def->isa<VariantIndex>()) {
        auto value = emit(vindex->op(0), bb);
        return bb->extract(convert(world().type_pu32())->type_id, value, { 0 });
    } else if (auto tuple = def->isa<Tuple>()) {
        std::vector<SpvId> elements;
        elements.resize(tuple->num_ops());
        size_t x = 0;
        for (auto& e : tuple->ops()) {
            elements[x++] = emit(e, bb);
        }
        return bb->composite(convert(tuple->type())->type_id, elements);
    } else if (auto structagg = def->isa<StructAgg>()) {
        std::vector<SpvId> elements;
        elements.resize(structagg->num_ops());
        size_t x = 0;
        for (auto& e : structagg->ops()) {
            elements[x++] = emit(e, bb);
        }
        return bb->composite(convert(structagg->type())->type_id, elements);
    } else if (auto access = def->isa<Access>()) {
        // emit dependent operations first
        emit(access->mem(), bb);

        std::vector<uint32_t> operands;
        auto ptr_type = access->ptr()->type()->as<PtrType>();
        if (ptr_type->addr_space() == AddrSpace::Global) {
            operands.push_back(spv::MemoryAccessAlignedMask);
            operands.push_back( 4 ); // TODO: SPIR-V docs say to consult client API for valid values.
        }
        if (auto load = def->isa<Load>()) {
            return bb->load(convert(load->out_val_type())->type_id, emit(load->ptr(), bb), operands);
        } else if (auto store = def->isa<Store>()) {
            bb->store(emit(store->val(), bb), emit(store->ptr(), bb), operands);
            return spv_none;
        } else THORIN_UNREACHABLE;
    } else if (auto lea = def->isa<LEA>()) {
        switch (lea->ptr_type()->addr_space()) {
            case AddrSpace::Global:
            case AddrSpace::Shared:
                break;
            default:
                world().ELOG("LEA is only allowed in global & shared address spaces");
                break;
        }
        auto type = convert(lea->ptr_type());
        auto offset = emit(lea->index(), bb);
        return bb->ptr_access_chain(type->type_id, emit(lea->ptr(), bb), offset, {});
    } else if (auto aggop = def->isa<AggOp>()) {
        auto spv_agg = emit(aggop->agg(), bb);
        auto agg_type = convert(aggop->agg()->type())->type_id;

        bool mem = false;
        if (auto tt = aggop->agg()->type()->isa<TupleType>(); tt && tt->op(0)->isa<MemType>()) mem = true;

        auto copy_to_alloca = [&] (SpvId target_type) {
            world().wdef(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto agg_ptr_type = builder_->declare_ptr_type(spv::StorageClassFunction, agg_type);

            auto variable = bb->fn_builder.variable(agg_ptr_type, spv::StorageClassFunction);
            bb->store(spv_agg, variable);

            auto cell_ptr_type = builder_->declare_ptr_type(spv::StorageClassFunction, target_type);
            auto cell = bb->access_chain(cell_ptr_type, variable, { emit(aggop->index(), bb)} );
            return std::make_pair(variable, cell);
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (is_mem(extract)) return spv_none;

            auto target_type = convert(extract->type())->type_id;
            auto constant_index = aggop->index()->isa<PrimLit>();

            // We have a fast-path: if the index is constant, we can simply use OpCompositeExtract
            if (aggop->agg()->type()->isa<ArrayType>() && constant_index == nullptr) {
                assert(aggop->agg()->type()->isa<DefiniteArrayType>());
                assert(!is_mem(extract));
                return bb->load(target_type, copy_to_alloca(target_type).second);
            }

            if (extract->agg()->type()->isa<VectorType>())
                return bb->vector_extract_dynamic(target_type, spv_agg, emit(extract->index(), bb));

            // index *must* be constant for the remaining possible cases
            assert(constant_index != nullptr);
            uint32_t index = constant_index->value().get_u32();

            unsigned offset = 0;
            if (mem) {
                if (aggop->agg()->type()->num_ops() == 2) return spv_agg;
                offset = 1;
            }

            return bb->extract(target_type, spv_agg, { index - offset });
        } else if (auto insert = def->isa<Insert>()) {
            auto value = emit(insert->value(), bb);
            auto constant_index = aggop->index()->isa<PrimLit>();

            // TODO deal with mem - but I think for now this case shouldn't happen

            if (insert->agg()->type()->isa<ArrayType>() && constant_index == nullptr) {
                assert(aggop->agg()->type()->isa<DefiniteArrayType>());
                auto [variable, cell] = copy_to_alloca(agg_type);
                bb->store(value, cell);
                return bb->load(agg_type, variable);
            }

            if (insert->agg()->type()->isa<VectorType>())
                return bb->vector_insert_dynamic(agg_type, spv_agg, value, emit(insert->index(), bb));

            // index *must* be constant for the remaining possible cases
            assert(constant_index != nullptr);
            uint32_t index = constant_index->value().get_u32();

            return bb->insert(agg_type, value, spv_agg, { index });
        } else THORIN_UNREACHABLE;
    } else if (auto conv = def->isa<ConvOp>()) {
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();

        auto conv_src_type = convert(src_type);
        auto conv_dst_type = convert(dst_type);

        if (auto bitcast = def->isa<Bitcast>()) {
            if (conv_src_type->datatype->serialized_size() != conv_dst_type->datatype->serialized_size())
                world().ELOG("Source (%) and destination (%) datatypes sizes do not match (% vs % bytes)", src_type->to_string(), dst_type->to_string(), conv_src_type->datatype->serialized_size(), conv_dst_type->datatype->serialized_size());

            return bb->convert(spv::OpBitcast, convert(bitcast->type())->type_id, emit(bitcast->from(), bb));
        } else if (auto cast = def->isa<Cast>()) {
            // NB: all ops used here are scalar/vector agnostic
            auto src_prim = src_type->isa<PrimType>();
            auto dst_prim = dst_type->isa<PrimType>();
            if (!src_prim || !dst_prim || src_prim->length() != dst_prim->length())
                world().ELOG("Illegal cast: % to %, casts are only supported between primitives with identical vector length", src_type->to_string(), dst_type->to_string());

            auto length = src_prim->length();

            auto src_kind = classify_primtype(src_prim);
            auto dst_kind = classify_primtype(dst_prim);
            size_t src_bitwidth = conv_src_type->datatype->serialized_size();
            size_t dst_bitwidth = conv_src_type->datatype->serialized_size();

            SpvId data = emit(cast->from(), bb);

            // If floating point is involved (src or dst), OpConvert*ToF and OpConvertFTo* can take care of the bit width transformation so no need for any chopping/expanding
            if (src_kind == PrimTypeKind::Float || dst_kind == PrimTypeKind::Float) {
                auto target_type = convert(get_primtype(world(), dst_kind, dst_bitwidth, length))->type_id;
                switch (src_kind) {
                    case PrimTypeKind::Signed:    data = bb->convert(spv::OpConvertSToF, target_type, data); break;
                    case PrimTypeKind::Unsigned:  data = bb->convert(spv::OpConvertUToF, target_type, data); break;
                    case PrimTypeKind::Float:
                        switch (dst_kind) {
                            case PrimTypeKind::Signed:   data = bb->convert(spv::OpConvertFToS, target_type, data); break;
                            case PrimTypeKind::Unsigned: data = bb->convert(spv::OpConvertFToU, target_type, data); break;
                            default: THORIN_UNREACHABLE;
                        }
                        break;
                }
            } else {
                // we expand first and shrink last to minimize precision losses, with bitcast in the middle
                bool needs_chopping = src_bitwidth > dst_bitwidth;
                bool needs_expanding = src_bitwidth < dst_bitwidth;

                if (needs_expanding) {
                    auto target_type = convert(get_primtype(world(), src_kind, src_bitwidth, length))->type_id;
                    switch (src_kind) {
                        case PrimTypeKind::Signed:
                            data = bb->convert(spv::OpSConvert, target_type, data);
                            break;
                        case PrimTypeKind::Unsigned:
                            data = bb->convert(spv::OpUConvert, target_type, data);
                            break;
                        case PrimTypeKind::Float:
                            data = bb->convert(spv::OpFConvert, target_type, data);
                            break;
                    }
                }

                auto expanded_bitwidth = needs_expanding ? dst_bitwidth : src_bitwidth;
                auto bitcast_target_type = convert(get_primtype(world(), dst_kind, expanded_bitwidth, length))->type_id;
                data = bb->convert(spv::OpBitcast, bitcast_target_type, data);

                if (needs_chopping) {
                    auto target_type = convert(get_primtype(world(), dst_kind, dst_bitwidth, length))->type_id;
                    switch (dst_kind) {
                        case PrimTypeKind::Signed:
                            data = bb->convert(spv::OpSConvert, target_type, data);
                            break;
                        case PrimTypeKind::Unsigned:
                            data = bb->convert(spv::OpUConvert, target_type, data);
                            break;
                        case PrimTypeKind::Float:
                            data = bb->convert(spv::OpFConvert, target_type, data);
                            break;
                    }
                }
            }
        } else THORIN_UNREACHABLE;
    } else if (def->isa<Bottom>()) {
        return bb->undef(convert(def->type())->type_id);
    }
    assertf(false, "Incomplete emit(def) definition");
}

std::vector<SpvId> CodeGen::emit_builtin(const Continuation* source_cont, const Continuation* builtin, BasicBlockBuilder* bb) {
    std::vector<SpvId> productions;
    auto uvec3_t = convert(world().type_pu32(3));
    auto u32_t = convert(world().type_pu32());
    auto i32_t = convert(world().type_ps32());
    if (builtin->name() == "spirv.nonsemantic.printf") {
        std::vector<SpvId> args;
        auto string = source_cont->arg(1);
        if (auto arr_type = string->type()->isa<DefiniteArrayType>(); arr_type->elem_type() == world().type_pu8()) {
            auto arr = string->as<DefiniteArray>();
            std::vector<char> the_string;
            for (size_t i = 0; i < arr_type->dim(); i++)
                the_string.push_back(arr->op(i)->as<PrimLit>()->value().get_u8());
            the_string.push_back('\0');
            args.push_back(builder_->debug_string(the_string.data()));
        } else world().ELOG("spirv.nonsemantic.printf takes a string literal");

        for (size_t i = 2; i < source_cont->num_args() - 1; i++) {
            args.push_back(emit(source_cont->arg(i), bb));
        }

        bb->ext_instruction(bb->file_builder.void_type, builder_->imported_instrs->shader_printf, 1, args);
    } else if (builtin->name() == "get_local_id") {
        auto vector = bb->load(uvec3_t->type_id, builder_->builtins->local_id);
        auto extracted = bb->vector_extract_dynamic(u32_t->type_id, vector, emit(source_cont->arg(1), bb));
        productions.push_back(bb->convert(spv::OpBitcast, i32_t->type_id, extracted));
    } else {
        world().ELOG("This spir-v builtin isn't recognised: %s", builtin->name());
    }
    return productions;
}

}
