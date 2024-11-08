#include "spirv_private.h"

#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/domtree.h"

#include <iostream>

namespace thorin::spirv {

/// Used as a dummy SSA value for emitting things like mem/unit
/// Should never make it in the binary files !
constexpr Id spv_none { 0 };

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
        : builder::BasicBlockBuilder(fn_builder.file_builder), fn_builder(fn_builder), file_builder(fn_builder.file_builder) {
    label = file_builder.generate_fresh_id();
}

FnBuilder::FnBuilder(FileBuilder& file_builder) : builder::FnBuilder(file_builder), file_builder(file_builder) {}

FileBuilder::FileBuilder(CodeGen* cg) : builder::FileBuilder(), cg(cg) {
}

Id FileBuilder::u32_t() {
    if (u32_t_ == 0)
        u32_t_ = cg->convert(cg->world().type_pu32()).id;
    return u32_t_;
}

Id FileBuilder::u32_constant(uint32_t pattern) {
    return constant(u32_t(), { pattern });
}

CodeGen::CodeGen(Thorin& thorin, Target& target_info, bool debug, const Cont2Config* kernel_config)
        : thorin::CodeGen(thorin, debug), target_info_(target_info), kernel_config_(kernel_config)
{}

void CodeGen::emit_stream(std::ostream& out) {
    FileBuilder builder(this);
    builder_ = &builder;

    switch (target_info_.dialect) {
        case Target::OpenCL:
            builder_->capability(spv::Capability::CapabilityKernel);
            builder_->capability(spv::Capability::CapabilityAddresses);
            builder_->addressing_model = target_info_.mem_layout.pointer_size == 4 ? spv::AddressingModelPhysical32 : spv::AddressingModelPhysical64;
            builder_->memory_model = spv::MemoryModel::MemoryModelOpenCL;
            builder_->version = 0x10200;
            break;
        case Target::Vulkan:
            builder_->capability(spv::Capability::CapabilityShader);
            builder_->addressing_model = spv::AddressingModelPhysicalStorageBuffer64;
            builder_->memory_model = spv::MemoryModel::MemoryModelGLSL450;
            break;
        default: assert(false && "unknown spirv dialect");
    }

    ScopesForest forest(world());
    forest.for_each([&](const Scope& scope) { emit_scope(scope, forest); });

    for (auto def : world().defs()) {
        if (auto global = def->isa<Global>())
            builder.interface.push_back(emit(global));
    }

    int entry_points_count = 0;
    for (auto& cont : world().copy_continuations()) {
        if (cont->is_exported() && kernel_config_) {
            auto config = kernel_config_->find(cont);
            if (config == kernel_config_->end())
                continue;

            assert(defs_.contains(cont));
            Id callee = defs_[cont];

            auto block = config->second->as<GPUKernelConfig>()->block_size();
            std::vector<uint32_t> local_size = {
                    (uint32_t) std::get<0>(block),
                    (uint32_t) std::get<1>(block),
                    (uint32_t) std::get<2>(block),
            };

            builder_->declare_entry_point(target_info_.dialect == Target::Vulkan ? spv::ExecutionModelGLCompute : spv::ExecutionModelKernel, callee, cont->name().c_str(), builder.interface);
            builder_->execution_mode(callee, spv::ExecutionModeLocalSize, local_size);
            entry_points_count++;
        }
    }

    if (entry_points_count == 0) {
        builder_->capability(spv::Capability::CapabilityLinkage);
    }

    builder_->finish(out);
    builder_ = nullptr;
}

Id CodeGen::emit_fun_decl(thorin::Continuation* continuation) {
    return get_fn_builder(continuation).function_id;
}

FnBuilder& CodeGen::get_fn_builder(thorin::Continuation* continuation) {
    if (auto found = builder_->fn_builders_.find(continuation); found != builder_->fn_builders_.end()) {
        return *found->second;
    }

    auto& fn = *(builder_->fn_builders_[continuation] = std::make_unique<FnBuilder>(*builder_));
    fn.fn_type = convert(entry_->type()).id;
    fn.fn_ret_type = get_codom_type(entry_);
    defs_[continuation] = fn.function_id;
    return fn;
}

FnBuilder* CodeGen::prepare(const thorin::Scope& scope) {
    auto& fn = get_fn_builder(scope.entry());
    builder_->name(fn.function_id, scope.entry()->name());
    builder_->current_fn_ = &fn;
    return &fn;
}

static bool is_return_block(thorin::Continuation* cont) {
    if (!cont->is_basicblock())
        return false;
    int uses_as_ret_param = 0;
    for (auto use : cont->copy_uses()) {
        if (use.def()->isa<Param>())
            continue; // the block can have params
        else if (auto app = use.def()->isa<App>()) {
            if (auto callee = app->callee()->isa_nom<Continuation>()) {
                auto arg_index = use.index() - App::FirstArg;
                auto ret_param = callee->ret_param();
                if (ret_param && arg_index == ret_param->index()) {
                    uses_as_ret_param++;
                    continue;
                }
            }
        }
        return false; // any other use disqualifies the block
    }
    return uses_as_ret_param == 1;
}

void CodeGen::prepare(thorin::Continuation* cont, FnBuilder* fn) {
    auto& bb = *fn->bbs.emplace_back(std::make_unique<BasicBlockBuilder>(*fn));
    cont2bb_[cont] = &bb;
    fn->bbs_to_emit.emplace_back(&bb);

    builder_->name(bb.label, cont->name().c_str());

    bb.semi_inline = is_return_block(cont);
    if (bb.semi_inline)
        world().ddef(cont, "Emitting {} as return block", cont);

    if (entry_ == cont) {
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param)) {
                // Nothing
                defs_[param] = 0;
            } else if (param->order() == 0) {
                auto param_t = convert(param->type());
                auto id = fn->parameter(param_t.id);
                defs_[param] = id;
                fn->params[param] = id;
                if (param->type()->isa<PtrType>()) {
                    builder_->decorate(id, spv::DecorationAliased);
                }
            }
        }
    } else {
        defs_[cont] = bb.label;
        if (bb.semi_inline)
            return;
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param)) {
                // Nothing
                defs_[param] = 0;
            } else {
                // OpPhi requires the full list of predecessors (values, labels)
                // We don't have that yet! But we will need the Phi node identifier to build the basic blocks ...
                // To solve this we generate an id for the phi node now, but defer emission of it to a later stage
                auto type = convert(param->type()).id;
                assert(type != 0);
                auto id = builder_->generate_fresh_id();
                defs_[param] = id;
                bb.phis_map[param] = { type, id, {} };
            }
        }
    }

}

void CodeGen::finalize(thorin::Continuation* cont) {
    auto& bb = *cont2bb_[cont];
    for (auto& [param, phi] : bb.phis_map) {
        bb.phis.emplace_back(&phi);
    }
}

void CodeGen::finalize(const thorin::Scope&) {
    builder_->define_function(*builder_->current_fn_);
}

Id CodeGen::get_codom_type(const Continuation* fn) {
    auto ret_cont_type = fn->ret_param()->type()->as<FnType>();
    std::vector<Id> types;
    for (auto& op : ret_cont_type->types()) {
        if (op->isa<MemType>() || is_type_unit(op->type()))
            continue;
        assert(op->order() == 0);
        types.push_back(convert(op).id);
    }
    if (types.empty())
        return convert(world().unit_type()).id;
    if (types.size() == 1)
        return types[0];
    return builder_->declare_struct_type(types);
}

Id CodeGen::emit_as_bb(thorin::Continuation* cont) {
    emit(cont);
    return cont2bb_[cont]->label;
}

void CodeGen::emit_epilogue(Continuation* continuation) {
    BasicBlockBuilder* bb = cont2bb_[continuation];

    // Handles the potential nuances of jumping to another continuation
    auto jump_to_next_cont_with_args = [&](Continuation* succ, std::vector<Id> args) {
        assert(succ->is_basicblock());
        BasicBlockBuilder* dstbb = cont2bb_[succ];

        for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
            assert(j < args.size());
            auto param = succ->param(i);
            if (is_mem(param) || is_unit(param)) {
                if (dstbb->semi_inline)
                    defs_[param] = 0;
                continue;
            }
            if (dstbb->semi_inline) {
                defs_[param] = args[j];
            } else {
                auto& phi = cont2bb_[succ]->phis_map[param];
                phi.preds.emplace_back(args[j], emit_as_bb(continuation));
            }
            j++;
        }
        bb->terminator.branch(emit(succ));
    };

    auto& app = *continuation->body();

    if (app.callee() == entry_->ret_param()) {
        std::vector<Id> values;

        for (auto arg : app.args()) {
            assert(arg->order() == 0);
            if (is_mem(arg) || is_unit(arg)) {
                emit_unsafe(arg);
                continue;
            }
            auto val = emit(arg);
            values.emplace_back(val);
        }

        switch (values.size()) {
            case 0:  bb->terminator.return_void();      break;
            case 1:  bb->terminator.return_value(values[0]); break;
            default: bb->terminator.return_value(emit_composite(bb, builder_->current_fn_->fn_ret_type, values));
        }
    } else if (auto dst_cont = app.callee()->isa_nom<Continuation>(); dst_cont && dst_cont->is_basicblock()) { // ordinary jump
        int index = -1;
        for (auto& arg : app.args()) {
            index++;
            if (is_mem(arg) || is_unit(arg)) {
                emit_unsafe(arg);
                continue;
            }
            auto val = emit(arg);
            auto* param = dst_cont->param(index);
            auto& phi = cont2bb_[dst_cont]->phis_map[param];
            phi.preds.emplace_back(val, emit_as_bb(continuation));
        }
        bb->terminator.branch(emit(dst_cont));
    } else if (app.callee() == world().branch()) {
        auto mem = app.arg(0);
        emit_unsafe(mem);

        auto cond = emit(app.arg(1));
        auto tbb = emit(app.arg(2));
        auto fbb = emit(app.arg(3));
        bb->terminator.branch_conditional(cond, tbb, fbb);
    } else if (app.callee()->isa<Continuation>() && app.callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
        emit_unsafe(app.arg(0));
        auto val = emit(app.arg(1));
        auto otherwise_bb = emit_as_bb(app.arg(2)->isa_nom<Continuation>());
        std::vector<Id> literals;
        std::vector<Id> cases;
        for (size_t i = 3; i < app.num_args(); i++) {
            auto arg = app.arg(i)->as<Tuple>();
            literals.push_back(emit(arg->op(0)));
            cases.push_back(emit_as_bb(arg->op(1)->as_nom<Continuation>()));
        }
        bb->terminator.branch_switch(val, otherwise_bb, literals, cases);
    } else if (app.callee()->isa<Bottom>()) {
        bb->terminator.unreachable();
    } else if (auto intrinsic = app.callee()->isa_nom<Continuation>(); intrinsic && (intrinsic->is_intrinsic() || intrinsic->cc() == CC::Device)) {
        // Ensure we emit previous memory operations
        assert(is_mem(app.arg(0)));
        emit_unsafe(app.arg(0));

        auto productions = emit_intrinsic(app, intrinsic, bb);
        auto succ = app.args().back()->isa_nom<Continuation>();
        jump_to_next_cont_with_args(succ, productions);
    } else { // function/closure call
        // put all first-order args into an array
        std::vector<Id> call_args;
        const Def* ret_arg = nullptr;
        for (auto arg : app.args()) {
            if (arg->order() == 0) {
                auto arg_type = arg->type();
                if (arg_type == world().unit_type() || arg_type == world().mem_type()) {
                    emit_unsafe(arg);
                    continue;
                }
                auto arg_val = emit(arg);
                call_args.push_back(arg_val);
            } else {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        Id call_result;
        if (auto called_continuation = app.callee()->isa_nom<Continuation>()) {
            auto ret_type = get_codom_type(called_continuation);
            call_result = bb->call(ret_type, emit(called_continuation), call_args);
        } else {
            // must be a closure
            THORIN_UNREACHABLE;

            // auto closure = emit(callee);
            // args.push_back(irbuilder.CreateExtractValue(closure, 1));
            // call = irbuilder.CreateCall(irbuilder.CreateExtractValue(closure, 0), args);
        }

        // must be call + continuation --- call + return has been removed by codegen_prepare
        auto succ = ret_arg->isa_nom<Continuation>();

        size_t real_params_count = 0;
        const Param* last_param = nullptr;
        for (auto param : succ->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            last_param = param;
            real_params_count++;
        }

        std::vector<Id> args(real_params_count);

        if (real_params_count == 1) {
            args[0] = call_result;
        } else if (real_params_count > 1) {
            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                args[j] = bb->extract(convert(param->type()).id, call_result, { (uint32_t) j });
                j++;
            }

            bb->terminator.branch(emit(succ));
        }

        jump_to_next_cont_with_args(succ, args);
    }
}

static_assert(sizeof(double) == sizeof(uint64_t), "This code assumes 64-bit double");

Id CodeGen::emit_constant(const thorin::Def* def) {
    if (auto primlit = def->isa<PrimLit>()) {
        Box box = primlit->value();
        auto type = convert(def->type()).id;
        Id constant;
        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     constant = builder_->bool_constant(type, box.get_bool()); break;
            case PrimType_ps8:  case PrimType_qs8:
            case PrimType_pu8:  case PrimType_qu8:  constant = builder_->constant(type, { static_cast<unsigned int>(box.get_u8()) }); break;
            case PrimType_ps16: case PrimType_qs16:
            case PrimType_pu16: case PrimType_qu16:
            case PrimType_pf16: case PrimType_qf16: constant = builder_->constant(type, { static_cast<unsigned int>(box.get_u16()) }); break;
            case PrimType_pf32: case PrimType_qf32:
            case PrimType_ps32: case PrimType_qs32:
            case PrimType_pu32: case PrimType_qu32: constant = builder_->constant(type, { static_cast<unsigned int>(box.get_u32()) }); break;
            case PrimType_pf64: case PrimType_qf64:
            case PrimType_ps64: case PrimType_qs64:
            case PrimType_pu64: case PrimType_qu64: {
                uint64_t value = static_cast<uint64_t>(box.get_u64());
                uint64_t upper = value >> 32U;
                uint64_t lower = value & 0xFFFFFFFFU;
                constant = builder_->constant(type, { (uint32_t) lower, (uint32_t) upper });
                break;
            }
        }
        return constant;
    }

    assertf(false, "Incomplete emit(def) definition");
}
Id CodeGen::emit_composite(BasicBlockBuilder* bb, Id t, Defs defs) {
    std::vector<Id> ids;
    for (auto& def : defs) {
        ids.push_back(emit(def));
    }
    return emit_composite(bb, t, ids);
}

Id CodeGen::emit_composite(BasicBlockBuilder* bb, Id t, ArrayRef<Id> ids) {
    if (target_info_.bugs.broken_op_construct_composite) {
        Id c = bb->undef(t);
        uint32_t x = 0;
        for (auto& e : ids) {
            c = bb->insert(t, e, c, { x++ });
        }
        return c;
    } else {
        std::vector<Id> elements;
        elements.resize(ids.size());
        size_t x = 0;
        for (auto& e : ids) {
            elements[x++] = e;
        }
        return bb->composite(t, elements);
    }
}

Id CodeGen::emit_bb(BasicBlockBuilder* bb, const Def* def) {
    if (auto mathop = def->isa<MathOp>())
        return emit_mathop(bb, *mathop);

    if (auto bin = def->isa<BinOp>()) {
        Id lhs = emit(bin->lhs());
        Id rhs = emit(bin->rhs());
        Id result_type = convert(def->type()).id;

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
    } else if (auto variant = def->isa<Variant>()) {
        assert(false && "TODO: rewrite");
        /*auto variant_type = def->type()->as<VariantType>();
        auto variant_datatype = (ProductDatatype*) convert(variant_type)->datatype.get();
        auto tag = builder_->u32_constant(variant->index());

        if (variant_datatype->elements_types.size() > 1) {
            auto alloc_type = convert(world().ptr_type(variant_datatype->elements_types[1]->src_type, 1, 4, AddrSpace::Function));
            auto payload_arr = current_fn_->variable(alloc_type, spv::StorageClassFunction);
            auto converted_payload_type = convert(variant_type->op(variant->index()));

            converted_payload_type->datatype->emit_serialization(*bb, spv::StorageClassFunction, payload_arr, bb->file_builder.u32_constant(0), emit(variant->value(), bb));
            auto payload = bb->load(variant_datatype->elements_types[1], payload_arr);

            std::vector<SpvId> with_tag = {tag, payload};
            return bb->composite(convert(variant->type()), with_tag);
        } else {
            // Zero-sized payload case
            std::vector<SpvId> with_tag = { tag };
            return bb->composite(convert(variant->type()), with_tag);
        }*/
    } else if (auto vextract = def->isa<VariantExtract>()) {
        assert(false && "TODO: rewrite");
        /*auto variant_type = vextract->value()->type()->as<VariantType>();
        auto variant_datatype = (ProductDatatype*) convert(variant_type)->datatype.get();

        auto target_type = convert(def->type());

        assert(variant_datatype->elements_types.size() > 1 && "Can't extract zero-sized datatypes");
        auto alloc_type = convert(world().ptr_type(variant_datatype->elements_types[1]->src_type, 1, 4, AddrSpace::Function));
        auto payload_arr = current_fn_->variable(alloc_type, spv::StorageClassFunction);
        auto payload = bb->extract(variant_datatype->elements_types[1], emit(vextract->value(), bb), {1});
        bb->store(payload, payload_arr);

        return target_type->datatype->emit_deserialization(*bb, spv::StorageClassFunction, payload_arr, bb->file_builder.u32_constant(0));*/
    } else if (auto vindex = def->isa<VariantIndex>()) {
        auto value = emit(vindex->op(0));
        return bb->extract(convert(world().type_pu32()).id, value, { 0 });
    } else if (auto tuple = def->isa<Tuple>()) {
        return emit_composite(bb, convert(tuple->type()).id, tuple->ops());
    } else if (auto structagg = def->isa<StructAgg>()) {
        return emit_composite(bb, convert(structagg->type()).id, structagg->ops());
    } else if (auto access = def->isa<Access>()) {
        // emit dependent operations first
        emit_unsafe(access->mem());

        std::vector<uint32_t> operands;
        auto ptr_type = access->ptr()->type()->as<PtrType>();
        if (ptr_type->addr_space() == AddrSpace::Global) {
            operands.push_back(spv::MemoryAccessAlignedMask);
            operands.push_back(4); // TODO: SPIR-V docs say to consult client API for valid values.
        }
        if (auto load = def->isa<Load>()) {
            return bb->load(convert(load->out_val_type()).id, emit(load->ptr()), operands);
        } else if (auto store = def->isa<Store>()) {
            bb->store(emit(store->val()), emit(store->ptr()), operands);
            return spv_none;
        } else THORIN_UNREACHABLE;
    } else if (auto slot = def->isa<Slot>()) {
        emit_unsafe(slot->frame());
        auto type = slot->type();
        auto id = bb->fn_builder.variable(convert(world().ptr_type(type->pointee(), 1, AddrSpace::Function)).id, spv::StorageClass::StorageClassFunction);
        id = bb->convert(spv::Op::OpPtrCastToGeneric, convert(type).id, id);
        return id;
    } else if (auto enter = def->isa<Enter>()) {
        return emit_unsafe(enter->mem());
    } else if (auto lea = def->isa<LEA>()) {
        //switch (lea->type()->addr_space()) {
        //    case AddrSpace::Global:
        //    case AddrSpace::Shared:
        //        break;
        //    default:
        //        world().ELOG("LEA is only allowed in global & shared address spaces");
        //        break;
        //}
        auto type = convert(lea->type()).id;
        auto offset = emit(lea->index());
        if (auto arr_type = lea->ptr_pointee()->isa<ArrayType>()) {
            auto base = bb->convert(spv::OpBitcast, type, emit(lea->ptr()));
            return bb->ptr_access_chain(type, base, offset, {  });
        }
        if (target_info_.bugs.static_ac_indices_must_be_i32)
            offset = emit(world().cast(world().type_pu32(), lea->index()));
        return bb->access_chain(type, emit(lea->ptr()), { offset });
    } else if (auto aggop = def->isa<AggOp>()) {
        auto agg_type = convert(aggop->agg()->type()).id;

        bool mem = false;
        if (auto tt = aggop->agg()->type()->isa<TupleType>(); tt && tt->op(0)->isa<MemType>()) mem = true;

        auto copy_to_alloca = [&] (Id spv_agg, Id target_type) {
            world().wdef(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto agg_ptr_type = builder_->declare_ptr_type(spv::StorageClassFunction, agg_type);

            auto variable = bb->fn_builder.variable(agg_ptr_type, spv::StorageClassFunction);
            bb->store(spv_agg, variable);

            auto cell_ptr_type = builder_->declare_ptr_type(spv::StorageClassFunction, target_type);
            auto cell = bb->access_chain(cell_ptr_type, variable, { emit(aggop->index())} );
            return std::make_pair(variable, cell);
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (is_mem(extract) || extract->type()->isa<FrameType>()) {
                emit_unsafe(extract->agg());
                return spv_none;
            }
            auto spv_agg = emit(aggop->agg());

            auto target_type = convert(extract->type()).id;
            auto constant_index = aggop->index()->isa<PrimLit>();

            // We have a fast-path: if the index is constant, we can simply use OpCompositeExtract
            if (aggop->agg()->type()->isa<ArrayType>() && constant_index == nullptr) {
                assert(aggop->agg()->type()->isa<DefiniteArrayType>());
                assert(!is_mem(extract));
                return bb->load(target_type, copy_to_alloca(spv_agg, target_type).second);
            }

            if (extract->agg()->type()->isa<VectorType>())
                return bb->vector_extract_dynamic(target_type, spv_agg, emit(extract->index()));

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
            auto spv_agg = emit(aggop->agg());
            auto value = emit(insert->value());
            auto constant_index = aggop->index()->isa<PrimLit>();

            // TODO deal with mem - but I think for now this case shouldn't happen

            if (insert->agg()->type()->isa<ArrayType>() && constant_index == nullptr) {
                assert(aggop->agg()->type()->isa<DefiniteArrayType>());
                auto [variable, cell] = copy_to_alloca(spv_agg, agg_type);
                bb->store(value, cell);
                return bb->load(agg_type, variable);
            }

            if (insert->agg()->type()->isa<VectorType>())
                return bb->vector_insert_dynamic(agg_type, spv_agg, value, emit(insert->index()));

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
            assert(conv_src_type.layout && conv_dst_type.layout);
            if (conv_src_type.layout->size != conv_dst_type.layout->size)
                world().ELOG("Source (%) and destination (%) datatypes sizes do not match (% vs % bytes)", src_type->to_string(), dst_type->to_string(), conv_src_type.layout->size, conv_dst_type.layout->size);

            return bb->convert(spv::OpBitcast, convert(bitcast->type()).id, emit(bitcast->from()));
        } else if (auto cast = def->isa<Cast>()) {
            // NB: all ops used here are scalar/vector agnostic
            auto src_prim = src_type->isa<PrimType>();
            auto dst_prim = dst_type->isa<PrimType>();
            if (!src_prim || !dst_prim || src_prim->length() != dst_prim->length())
                world().ELOG("Illegal cast: % to %, casts are only supported between primitives with identical vector length", src_type->to_string(), dst_type->to_string());

            auto length = src_prim->length();

            auto src_kind = classify_primtype(src_prim);
            auto dst_kind = classify_primtype(dst_prim);
            size_t src_bitwidth = conv_src_type.layout->size * 8;
            size_t dst_bitwidth = conv_dst_type.layout->size * 8;

            Id data = emit(cast->from());

            // If floating point is involved (src or dst), OpConvert*ToF and OpConvertFTo* can take care of the bit width transformation so no need for any chopping/expanding
            if (src_kind == PrimTypeKind::Float || dst_kind == PrimTypeKind::Float) {
                auto target_type = convert(get_primtype(world(), dst_kind, dst_bitwidth, length)).id;
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
                    auto target_type = convert(get_primtype(world(), src_kind, src_bitwidth, length)).id;
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
                        default: assert(false);
                    }
                }

                auto expanded_bitwidth = needs_expanding ? dst_bitwidth : src_bitwidth;
                auto bitcast_target_type = convert(get_primtype(world(), dst_kind, expanded_bitwidth, length)).id;
                data = bb->convert(spv::OpBitcast, bitcast_target_type, data);

                if (needs_chopping) {
                    auto target_type = convert(get_primtype(world(), dst_kind, dst_bitwidth, length)).id;
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
                        default: assert(false);
                    }
                }
            }

            return data;
        } else THORIN_UNREACHABLE;
    } else if (def->isa<Bottom>()) {
        return bb->undef(convert(def->type()).id);
    }

    if (!def->has_dep(Dep::Param))
        return emit_constant(def);

    assertf(false, "Incomplete emit(def) definition");
}

}
