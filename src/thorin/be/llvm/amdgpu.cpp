#include "thorin/be/llvm/amdgpu.h"

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

AMDGPUCodeGen::AMDGPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_KERNEL)
{
    module_->setDataLayout("e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64");
    module_->setTargetTriple("amdgcn-amd-amdhsa-amdgiz");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

llvm::Value* AMDGPUCodeGen::emit_global(const Global* global) {
    WLOG(global, "AMDGPU: Global variable '{}' will not be synced with host.", global);
    return CodeGen::emit_global(global);
}

Continuation* AMDGPUCodeGen::emit_reserve(const Continuation* continuation) { return emit_reserve_shared(continuation, true); }

}
