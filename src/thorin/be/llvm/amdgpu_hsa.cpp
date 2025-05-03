#include "thorin/be/llvm/amdgpu_hsa.h"

namespace thorin::llvm {

AMDGPUHSACodeGen::AMDGPUHSACodeGen(Thorin& thorin, const Cont2Config& kernel_config, int opt, bool debug)
    : AMDGPUCodeGen(thorin, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_KERNEL, kernel_config, opt, debug)
{
    module().setDataLayout("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");
    module().setTargetTriple("amdgcn-amd-amdhsa");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

llvm::Function* AMDGPUHSACodeGen::emit_fun_decl(Continuation* continuation) {
    if (continuation->name() == "llvm.amdgcn.implicitarg.ptr")
        if (auto f = defs_.lookup(entry_); f && llvm::isa<llvm::Function>(*f))
            llvm::cast<llvm::Function>(*f)->addFnAttr("amdgpu-implicitarg-ptr");
    if (continuation->name() == "__ockl_printf_begin")
        if (auto f = defs_.lookup(entry_); f && llvm::isa<llvm::Function>(*f))
            llvm::cast<llvm::Function>(*f)->addFnAttr("amdgpu-implicitarg-num-bytes", "32");
    return CodeGen::emit_fun_decl(continuation);
}

}
