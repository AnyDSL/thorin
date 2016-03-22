#include "thorin/be/llvm/spir.h"

#include <llvm/ADT/Triple.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

SPIRCodeGen::SPIRCodeGen(World& world)
    : CodeGen(world, llvm::Function::ExternalLinkage, llvm::Function::ExternalLinkage, llvm::CallingConv::SPIR_FUNC, llvm::CallingConv::SPIR_FUNC, llvm::CallingConv::SPIR_KERNEL)
{
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    if (triple.isArch32Bit()) {
        module_->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024");
        module_->setTargetTriple("spir-unknown-unknown");
    } else {
        module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024");
        module_->setTargetTriple("spir64-unknown-unknown");
    }
    // opencl.enable.FP_CONTRACT
    module_->getOrInsertNamedMetadata("opencl.enable.FP_CONTRACT");
    // opencl.spir.version
    auto spir_version_md = module_->getOrInsertNamedMetadata("opencl.spir.version");
    llvm::Value* annotation_values_12[] = { irbuilder_.getInt32(1), irbuilder_.getInt32(2) };
    spir_version_md->addOperand(llvm::MDNode::get(context_, annotation_values_12));
    // opencl.ocl.version
    auto ocl_version_md = module_->getOrInsertNamedMetadata("opencl.ocl.version");
    ocl_version_md->addOperand(llvm::MDNode::get(context_, annotation_values_12));
    // opencl.used.extensions
    module_->getOrInsertNamedMetadata("opencl.used.extensions");
    // opencl.used.optional.core.features
    module_->getOrInsertNamedMetadata("opencl.used.optional.core.features");
    // opencl.compiler.options
    module_->getOrInsertNamedMetadata("opencl.compiler.options");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

void SPIRCodeGen::emit_function_decl_hook(Continuation* continuation, llvm::Function* f) {
    // append required metadata
    size_t num_params = f->arg_size() + 1;
    Array<llvm::Value*> annotation_values_addr_space(num_params);
    Array<llvm::Value*> annotation_values_access_qual(num_params);
    Array<llvm::Value*> annotation_values_type(num_params);
    Array<llvm::Value*> annotation_values_type_qual(num_params);
    Array<llvm::Value*> annotation_values_name(num_params);
    annotation_values_addr_space[0]  = llvm::MDString::get(context_, "kernel_arg_addr_space");
    annotation_values_access_qual[0] = llvm::MDString::get(context_, "kernel_arg_access_qual");
    annotation_values_type[0]        = llvm::MDString::get(context_, "kernel_arg_type");
    annotation_values_type_qual[0]   = llvm::MDString::get(context_, "kernel_arg_type_qual");
    annotation_values_name[0]        = llvm::MDString::get(context_, "kernel_arg_name");
    size_t index = 1;
    for (auto it = f->arg_begin(), e = f->arg_end(); it != e; ++it, ++index) {
        llvm::Type* type = it->getType();
        size_t addr_space = 0;
        if (llvm::isa<llvm::PointerType>(type)) {
            addr_space = llvm::dyn_cast<llvm::PointerType>(type)->getAddressSpace();
            type = llvm::dyn_cast<llvm::PointerType>(type)->getElementType()->getPointerTo(0);
        }
        annotation_values_addr_space[index] = irbuilder_.getInt32(addr_space);
        annotation_values_access_qual[index] = llvm::MDString::get(context_, "none");
        std::string type_string;
        llvm::raw_string_ostream type_os(type_string);
        type->print(type_os);
        type_os.flush();
        annotation_values_type[index] = llvm::MDString::get(context_, type_string);
        annotation_values_type_qual[index] = llvm::MDString::get(context_, "");
        annotation_values_name[index] = llvm::MDString::get(context_, continuation->param(index + 1)->name);
    }
    llvm::Value* annotation_values_kernel[] = {
        f,
        llvm::MDNode::get(context_, llvm_ref(annotation_values_addr_space)),
        llvm::MDNode::get(context_, llvm_ref(annotation_values_access_qual)),
        llvm::MDNode::get(context_, llvm_ref(annotation_values_type)),
        llvm::MDNode::get(context_, llvm_ref(annotation_values_type_qual)),
        llvm::MDNode::get(context_, llvm_ref(annotation_values_name))
    };
    // opencl.kernels
    auto kernels_md = module_->getOrInsertNamedMetadata("opencl.kernels");
    kernels_md->addOperand(llvm::MDNode::get(context_, annotation_values_kernel));
}

Continuation* SPIRCodeGen::emit_reserve(const Continuation* continuation) { return emit_reserve_shared(continuation, true /* add kernel prefix */); }

}
