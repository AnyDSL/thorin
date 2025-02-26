#include "thorin/be/llvm/amdgpu_pal.h"

namespace thorin::llvm {

AMDGPUPALCodeGen::AMDGPUPALCodeGen(Thorin& thorin, const Cont2Config& kernel_config, int opt, bool debug)
    : AMDGPUCodeGen(thorin, llvm::CallingConv::AMDGPU_Gfx, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_CS, kernel_config, opt, debug)
{
    module().setDataLayout("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");
    module().setTargetTriple("amdgcn-amd-amdpal");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------


llvm::Function* AMDGPUPALCodeGen::emit_fun_decl(Continuation* continuation) {
    return CodeGen::emit_fun_decl(continuation);
}

}
