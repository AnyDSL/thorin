#include "thorin/be/llvm/amdgpu.h"

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

AMDGPUCodeGen::AMDGPUCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_KERNEL)
    , kernel_config_(kernel_config)
{
    module_->setDataLayout("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5");
    module_->setTargetTriple("amdgcn-amd-amdhsa");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

void AMDGPUCodeGen::emit_function_decl_hook(Continuation* continuation, llvm::Function* f) {
    auto config = kernel_config_.find(continuation);
    if (config != kernel_config_.end()) {
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            Array<llvm::Metadata*> annotation_values_wgsize(3);
            annotation_values_wgsize[0] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<0>(block)));
            annotation_values_wgsize[1] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<1>(block)));
            annotation_values_wgsize[2] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<2>(block)));
            f->setMetadata(llvm::StringRef("reqd_work_group_size"),  llvm::MDNode::get(*context_, llvm_ref(annotation_values_wgsize)));
        }
    }
}

llvm::Value* AMDGPUCodeGen::emit_global(const Global* global) {
    if (global->is_mutable())
        WDEF(global, "AMDGPU: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

Continuation* AMDGPUCodeGen::emit_reserve(const Continuation* continuation) { return emit_reserve_shared(continuation, true); }

llvm::SyncScope::ID AMDGPUCodeGen::get_atomic_sync_scope(const AddrSpace addr_space) const {
    switch (addr_space) {
        case AddrSpace::Generic:
        case AddrSpace::Global:
        case AddrSpace::Texture:
        case AddrSpace::Constant: return context_->getOrInsertSyncScopeID("agent");
        case AddrSpace::Shared:   return context_->getOrInsertSyncScopeID("workgroup");;
        default:                  THORIN_UNREACHABLE;
    }
}

}
