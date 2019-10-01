#include "thorin/be/llvm/amdgpu.h"

#include "thorin/world.h"

namespace thorin {

AMDGPUCodeGen::AMDGPUCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::AMDGPU_KERNEL)
    , kernel_config_(kernel_config)
{
    module_->setDataLayout("e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64");
    module_->setTargetTriple("amdgcn-amd-amdhsa");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

void AMDGPUCodeGen::emit_function_decl_hook(Lam* lam, llvm::Function* f) {
    auto config = kernel_config_.find(lam);
    if (config != kernel_config_.end()) {
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            Array<llvm::Metadata*> annotation_values_wgsize(3);
            annotation_values_wgsize[0] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<0>(block)));
            annotation_values_wgsize[1] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<1>(block)));
            annotation_values_wgsize[2] = llvm::ConstantAsMetadata::get(irbuilder_.getInt32(std::get<2>(block)));
            f->setMetadata(llvm::StringRef("reqd_work_group_size"),  llvm::MDNode::get(context_, llvm_ref(annotation_values_wgsize)));
        }
    }
}

unsigned AMDGPUCodeGen::convert_addr_space(u64 addr_space) {
    switch (addr_space) {
        case AddrSpace::Generic:
        case AddrSpace::Global:   return 1;
        case AddrSpace::Texture:  return 2;
        case AddrSpace::Shared:   return 3;
        case AddrSpace::Constant: return 4;
        default:                  THORIN_UNREACHABLE;
    }
}

llvm::Value* AMDGPUCodeGen::emit_global(const Global* global) {
    world().wdef(global, "AMDGPU: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

Lam* AMDGPUCodeGen::emit_reserve(Lam* lam) { return emit_reserve_shared(lam, true); }

}
