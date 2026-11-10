#include "spirv_private.h"

namespace thorin::spirv {

struct SpirMathOps {
#define THORIN_MATHOP(mathop_name) builder::ExtendedInstruction mathop_name;
#include "thorin/tables/mathoptable.h"
#undef THORIN_MATHOP
};

#include <spirv/unified1/OpenCL.std.h>

SpirMathOps opencl_std = {
    .fabs =     { "OpenCL.std", OpenCLLIB::Fabs },
    .copysign = { "OpenCL.std", OpenCLLIB::Copysign },
    .round =    { "OpenCL.std", OpenCLLIB::Round },
    .floor =    { "OpenCL.std", OpenCLLIB::Floor },
    .ceil =     { "OpenCL.std", OpenCLLIB::Ceil },
    .fmin =     { "OpenCL.std", OpenCLLIB::Fmin },
    .fmax =     { "OpenCL.std", OpenCLLIB::Fmax },
    .cos =      { "OpenCL.std", OpenCLLIB::Cos },
    .sin =      { "OpenCL.std", OpenCLLIB::Sin },
    .tan =      { "OpenCL.std", OpenCLLIB::Tan },
    .acos =     { "OpenCL.std", OpenCLLIB::Acos },
    .asin =     { "OpenCL.std", OpenCLLIB::Asin },
    .atan =     { "OpenCL.std", OpenCLLIB::Atan },
    .atan2 =    { "OpenCL.std", OpenCLLIB::Atan2 },
    .sqrt =     { "OpenCL.std", OpenCLLIB::Sqrt },
    .cbrt =     { "OpenCL.std", OpenCLLIB::Cbrt },
    .pow =      { "OpenCL.std", OpenCLLIB::Pow },
    .exp =      { "OpenCL.std", OpenCLLIB::Exp },
    .exp2 =     { "OpenCL.std", OpenCLLIB::Exp2 },
    .log =      { "OpenCL.std", OpenCLLIB::Log },
    .log2 =     { "OpenCL.std", OpenCLLIB::Log2 },
    .log10 =    { "OpenCL.std", OpenCLLIB::Log10 },
};

Id CodeGen::emit_mathop(BasicBlockBuilder* bb, const thorin::MathOp& mathop) {
    auto type = mathop.type();

    SpirMathOps& impl = opencl_std;
    std::vector<Id> ops;
    for (auto& op : mathop.ops()) {
        ops.push_back(emit(op));
    }

    if (is_type_f(type)) {
        switch (mathop.mathop_tag()) {
#define THORIN_MATHOP(mathop_name) case MathOp_##mathop_name: return bb->ext_instruction(convert(type).id, impl.mathop_name, ops);
#include "thorin/tables/mathoptable.h"
        }
    }
}

std::tuple<spv::Scope, spv::MemorySemanticsMask> addrspace_atomics_params(World& world, AddrSpace as) {
    switch (as) {
        case AddrSpace::Global:
            return std::make_tuple(spv::ScopeDevice, spv::MemorySemanticsMask::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsMask::MemorySemanticsCrossWorkgroupMemoryMask);
        case AddrSpace::Shared:
            return std::make_tuple(spv::ScopeWorkgroup, spv::MemorySemanticsMask::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsMask::MemorySemanticsWorkgroupMemoryMask);
        default:
            world.ELOG("Unsupported address space for atomics: {}", (int) as);
            THORIN_UNREACHABLE;
    }
}

std::vector<Id> CodeGen::emit_intrinsic(const App& app, const Continuation* intrinsic, BasicBlockBuilder* bb) {
    auto get_produced_types = [&]() {
        auto fn_type = intrinsic->type();
        return *(fn_type->codomain());
    };

    auto get_produced_type = [&]() {
        return get_produced_types()[1];
    };

    SpirMathOps& impl = opencl_std;
    auto intrinsic_name = intrinsic->name();
#define THORIN_MATHOP(mathop_name) \
    if ((#mathop_name) == intrinsic_name) { \
        return { bb->ext_instruction(convert(get_produced_type()).id, impl.mathop_name, emit_args(app.args().skip_back())) }; \
    }
#include "thorin/tables/mathoptable.h"

    if (intrinsic->name() == "spirv.nonsemantic.printf") {
        std::vector<Id> args;
        auto string = app.arg(1);
        if (auto arr_type = string->type()->isa<DefiniteArrayType>(); arr_type->elem_type() == world().type_pu8()) {
            auto arr = string->as<DefiniteArray>();
            std::vector<char> the_string;
            for (size_t i = 0; i < arr_type->dim(); i++)
                the_string.push_back(arr->op(i)->as<PrimLit>()->value().get_u8());
            the_string.push_back('\0');
            args.push_back(builder_->debug_string(the_string.data()));
        } else world().ELOG("spirv.nonsemantic.printf takes a string literal");

        for (size_t i = 2; i < app.num_args() - 1; i++) {
            args.push_back(emit(app.arg(i)));
        }

        builder_->extension("SPV_KHR_non_semantic_info");
        bb->ext_instruction(convert(world().unit_type()).id, { "NonSemantic.DebugPrintf", 1}, args);
        return {};

    } else if (intrinsic->name() == "spirv.builtin") {
        std::vector<Id> productions;
        if (auto spv_builtin_lit = app.arg(1)->isa<PrimLit>()) {
            auto spv_builtin = spv_builtin_lit->value().get_u32();
            auto found = builder_->builtins_.find(spv_builtin);
            if (found != builder_->builtins_.end()) {
                productions.push_back(found->second);
            } else {
                auto desired_type = get_produced_type()->as<PtrType>();
                auto id = builder_->global_variable(convert(desired_type).id, static_cast<spv::StorageClass>(convert(desired_type->addr_space())));
                builder_->interface.push_back(id);
                builder_->decorate(id, spv::Decoration::DecorationBuiltIn, { spv_builtin });
                builder_->builtins_[spv_builtin] = id;
                productions.push_back(id);
            }
            return productions;
        } else
            world().ELOG("spirv.builtin requires an integer literal as the argument");

    } else if (intrinsic->name() == "spirv.operation") {
        if (auto spv_operation_lit = app.arg(1)->isa<PrimLit>()) {
            auto spv_operation = spv_operation_lit->value().get_u32();
            std::vector<unsigned int> args;
            if (app.arg(2)->type()->isa<TupleType>()) {
                args = emit_args(app.arg(2)->ops());
            } else {
                args = emit_args({ app.arg(2) });
            }

            spv::Op op = static_cast<spv::Op>(spv_operation);
            auto produced_types = get_produced_types();
            if (produced_types.size() > 1) {
                auto result = bb->op_with_result(op, convert(produced_types[1]).id, args);
                return { result };
            } else {
                bb->op(op, args);
                return { };
            }
        } else
            world().ELOG("spirv.operation requires an integer literal as the first argument");

    } else if (intrinsic->name() == "spirv.literal") {
        if (auto spv_lit = app.arg(1)->isa<PrimLit>()) {
            auto literal = spv_lit->value().get_u32();
            return { literal };
        } else
            world().ELOG("spirv.literal requires an integer literal as the first argument");

    } else if (intrinsic->name() == "spirv.extension") {
        auto string = app.arg(1)->op(0)->op(0); //TODO: this can fail!
        if (auto arr_type = string->type()->isa<DefiniteArrayType>(); arr_type->elem_type() == world().type_pu8()) {
            auto arr = string->as<DefiniteArray>();
            std::vector<char> the_string;
            for (size_t i = 0; i < arr_type->dim(); i++)
                the_string.push_back(arr->op(i)->as<PrimLit>()->value().get_u8());

            std::string extension(the_string.begin(), the_string.end());

            builder_->extension(extension);

            return { };
        } else
            world().ELOG("spirv.extension requires a string literal as the argument");

    } else if (intrinsic->name() == "spirv.capability") {
        if (auto spv_capability_lit = app.arg(1)->isa<PrimLit>()) {
            auto spv_capability = spv_capability_lit->value().get_u32();

            builder_->capability(static_cast<spv::Capability>(spv_capability));

            return { };
        } else
            world().ELOG("spirv.capability requires an integer literal as the argument");

    } else if (intrinsic->name() == "reserve_shared") {
        auto produced_t = get_produced_type()->as<PtrType>();
        auto pointee = produced_t->pointee();
        if (auto indef = pointee->isa<IndefiniteArrayType>())
            pointee = indef->elem_type();
        auto size = app.arg(1)->isa<PrimLit>();
        if (!size)
            world().error(app.loc(), "reserve_shared called with non-constant size");
        else {
            auto in_bytes = size->value().get_u64() * convert(pointee).layout->size;
            auto type = world().definite_array_type(world().type_pu8(), in_bytes);
            Id id = builder_->global_variable(convert(world().ptr_type(type, 1, AddrSpace::Shared)).id, static_cast<spv::StorageClass>(convert(AddrSpace::Shared)));
            id = bb->convert(spv::Op::OpBitcast, convert(produced_t).id, id);
            return { id };
        }
    } else if (intrinsic->name() == "min") {
        auto type = get_produced_type();
        if (is_type_f(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::Fmin }, emit_args(app.args().skip_back())) };
        if (is_type_u(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::UMin }, emit_args(app.args().skip_back())) };
        if (is_type_i(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::SMin }, emit_args(app.args().skip_back())) };
    } else if (intrinsic->name() == "max") {
        auto type = get_produced_type();
        if (is_type_f(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::Fmax }, emit_args(app.args().skip_back())) };
        if (is_type_u(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::UMax }, emit_args(app.args().skip_back())) };
        if (is_type_i(type))
            return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::SMax }, emit_args(app.args().skip_back())) };
    } else if (intrinsic->name() == "barrier") {
        emit_args(app.args().skip_back());
        bb->op(spv::Op::OpControlBarrier, { literal(spv::Scope::ScopeWorkgroup), literal(spv::Scope::ScopeWorkgroup), literal(spv::MemorySemanticsMask::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsMask::MemorySemanticsSequentiallyConsistentMask) });
        return { };
    } else if (intrinsic->name() == "atomic_add") {
        auto args = emit_args(app.args().skip_back());
        auto [ptr, value] = *(std::array<Id, 2>*)args.data();
        auto produced = get_produced_type();
        spv::Op op;
        if (is_type_f(produced)) {
            op = spv::OpAtomicFAddEXT;
            auto ct = convert(produced);
            switch (ct.layout->size) {
                case 2: builder_->capability(spv::Capability::CapabilityAtomicFloat16AddEXT); break;
                case 4: builder_->capability(spv::Capability::CapabilityAtomicFloat32AddEXT); break;
                case 8: builder_->capability(spv::Capability::CapabilityAtomicFloat64AddEXT); break;
            }
            builder_->extension("SPV_EXT_shader_atomic_float_add");
        } else if (is_type_i(produced))
            op = spv::OpAtomicIAdd;
        else
            assert(false && "unknown primitive type for atomic_add");
        auto [scope, semantics] = addrspace_atomics_params(world(), app.arg(1)->type()->as<PtrType>()->addr_space());
        auto result = bb->op_with_result(op, convert(get_produced_type()).id,  { ptr, literal(scope), literal(semantics), value });
        return { result };
    } else if (intrinsic->name() == "atomic_min") {
        auto args = emit_args(app.args().skip_back());
        auto [ptr, value] = *(std::array<Id, 2>*)args.data();
        auto produced = get_produced_type();
        spv::Op op;
        if (is_type_f(produced)) {
            op = spv::OpAtomicFMinEXT;
            auto ct = convert(produced);
            switch (ct.layout->size) {
                case 2: builder_->capability(spv::Capability::CapabilityAtomicFloat16MinMaxEXT); break;
                case 4: builder_->capability(spv::Capability::CapabilityAtomicFloat32MinMaxEXT); break;
                case 8: builder_->capability(spv::Capability::CapabilityAtomicFloat64MinMaxEXT); break;
            }
            builder_->extension("SPV_EXT_shader_atomic_float_min_max");
        } else if (is_type_s(produced))
            op = spv::OpAtomicSMin;
        else if (is_type_u(produced))
            op = spv::OpAtomicUMin;
        else
            assert(false && "unknown primitive type for atomic_add");
        auto [scope, semantics] = addrspace_atomics_params(world(), app.arg(1)->type()->as<PtrType>()->addr_space());
        auto result = bb->op_with_result(op, convert(get_produced_type()).id,  { ptr, literal(scope), literal(semantics), value });
        return { result };
    } else if (intrinsic->name() == "rv_all") {
        auto args = emit_args(app.args().skip_back());
        auto result = bb->op_with_result(spv::Op::OpGroupAll, convert(get_produced_type()).id,  { literal(spv::Scope::ScopeInvocation), emit(app.arg(1)) });
        return { result };
    } else if (intrinsic->name() == "spirv.printf") {
        Array<const Def*> args = app.args().skip_back();
        while (auto bitcast = args[1]->isa<Bitcast>())
            args[1] = bitcast->from();
        return { bb->ext_instruction(convert(get_produced_type()).id, { .set_name = "OpenCL.std", .id = OpenCLLIB::Printf }, emit_args(args)) };

    } else if (intrinsic->name() == "gpu_set_subgroup_size") {
        return { };

    }
    world().ELOG("thorin/spirv: Intrinsic '{}' isn't recognised", intrinsic->name());
    exit(-1);
}

}
