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

std::vector<Id> CodeGen::emit_intrinsic(const App& app, const Continuation* intrinsic, BasicBlockBuilder* bb) {
    std::vector<Id> productions;

    auto get_produced_type = [&]() {
        auto ret_type = (*intrinsic->params().back()).type()->as<FnType>();
        return ret_type->types()[1];
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
    } else if (intrinsic->name() == "spirv.builtin") {
        if (auto spv_builtin_lit = app.arg(1)->isa<PrimLit>()) {
            auto spv_builtin = spv_builtin_lit->value().get_u32();
            auto found = builder_->builtins_.find(spv_builtin);
            if (found != builder_->builtins_.end()) {
                productions.push_back(found->second);
            } else {
                auto desired_type = get_produced_type()->as<PtrType>();
                auto id = builder_->variable(convert(desired_type).id, static_cast<spv::StorageClass>(convert(desired_type->addr_space())));
                builder_->interface.push_back(id);
                builder_->decorate(id, spv::Decoration::DecorationBuiltIn, { spv_builtin });
                builder_->builtins_[spv_builtin] = id;
                productions.push_back(id);
            }
        } else
            world().ELOG("spirv.builtin requires an integer literal as the argument");
    } else {
        world().ELOG("thorin/spirv: Intrinsic '{}' isn't recognised", intrinsic->name());
    }
    return productions;
}

}
