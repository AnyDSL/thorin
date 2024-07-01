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

SpvId CodeGen::emit_mathop(BasicBlockBuilder* bb, const thorin::MathOp& mathop) {
    auto type = mathop.type();

    SpirMathOps& impl = opencl_std;
    std::vector<SpvId> ops;
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

}