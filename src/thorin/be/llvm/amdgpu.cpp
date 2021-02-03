#include "thorin/be/llvm/amdgpu.h"

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

AMDGPUCodeGen::AMDGPUCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_KERNEL, opt, debug)
    , kernel_config_(kernel_config)
{
    module().setDataLayout("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7");
    module().setTargetTriple("amdgcn-amd-amdhsa");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

void AMDGPUCodeGen::emit_function_decl_hook(Continuation* continuation, llvm::Function* f) {
    auto config = kernel_config_.find(continuation);
    if (config != kernel_config_.end()) {
        auto& irbuilder = *cont2llvm_[continuation]->second;
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            Array<llvm::Metadata*> annotation_values_wgsize(3);
            annotation_values_wgsize[0] = llvm::ConstantAsMetadata::get(irbuilder.getInt32(std::get<0>(block)));
            annotation_values_wgsize[1] = llvm::ConstantAsMetadata::get(irbuilder.getInt32(std::get<1>(block)));
            annotation_values_wgsize[2] = llvm::ConstantAsMetadata::get(irbuilder.getInt32(std::get<2>(block)));
            f->setMetadata(llvm::StringRef("reqd_work_group_size"), llvm::MDNode::get(context(), llvm_ref(annotation_values_wgsize)));
        }
    }
}

llvm::Function* AMDGPUCodeGen::emit_function_decl(Continuation* continuation) {
    if (continuation->name() == "llvm.amdgcn.implicitarg.ptr")
        if (auto f = def2llvm_.lookup(entry_); f && llvm::isa<llvm::Function>(*f))
            llvm::cast<llvm::Function>(*f)->addFnAttr("amdgpu-implicitarg-ptr");
    if (continuation->name() == "__ockl_printf_begin")
        if (auto f = def2llvm_.lookup(entry_); f && llvm::isa<llvm::Function>(*f))
            llvm::cast<llvm::Function>(*f)->addFnAttr("amdgpu-implicitarg-num-bytes", "32");
    return CodeGen::emit_function_decl(continuation);
}

llvm::Value* AMDGPUCodeGen::emit_global(const Global* global) {
    if (global->is_mutable())
        world().wdef(global, "AMDGPU: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

Continuation* AMDGPUCodeGen::emit_reserve(llvm::IRBuilder<>& irbuilder, const Continuation* continuation) {
    return emit_reserve_shared(irbuilder, continuation, true);
}

}
