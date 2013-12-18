#include "thorin/be/llvm/nvvm.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

llvm::Function* NVVMCodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "kernel", module_);
    f->setCallingConv(llvm::CallingConv::PTX_Kernel);
    // append required metadata
    llvm::NamedMDNode* annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Value* annotation_values[] = {
        f,
        llvm::MDString::get(context_, lambda->unique_name()),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), 1)
    };
    annotation->addOperand(llvm::MDNode::get(context_, annotation_values));
    return f;
}

Lambda* CodeGen::emit_nvvm(Lambda* lambda) {
    // to-target is the desired cuda call
    // target(mem, size, body, return, free_vars)
    Lambda* target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::NVVM));
    assert(lambda->num_args() > 3 && "required arguments are missing");
    // get input
    const uint64_t it_space_x = lambda->arg(1)->as<PrimLit>()->u64_value();
    Lambda* kernel = lambda->arg(2)->as_lambda();
    Lambda* ret = lambda->arg(3)->as_lambda();

    // load kernel
    llvm::Value* module_name = builder_.CreateGlobalStringPtr(kernel->unique_name());
    llvm::Value* kernel_name = builder_.CreateGlobalStringPtr("kernel");
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(llvm_decls_.get_nvvm_load_kernel(), load_args);
    // fetch values and create external calls for intialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = cuda_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        llvm::Constant* size = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), num_elems);
        auto alloca = builder_.CreateAlloca(llvm_decls_.get_nvvm_device_ptr_type());
        auto device_ptr = builder_.CreateCall(llvm_decls_.get_nvvm_malloc_memory(), size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::pair<llvm::Value*, llvm::Constant*>(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), size };
        builder_.CreateCall(llvm_decls_.get_nvvm_write_memory(), mem_args);
        builder_.CreateCall(llvm_decls_.get_nvvm_set_kernel_arg(), { alloca });
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), it_space_x),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), 1),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), 1)
    };
    builder_.CreateCall(llvm_decls_.get_nvvm_set_problem_size(), problem_size_args);
    // launch
    builder_.CreateCall(llvm_decls_.get_nvvm_launch_kernel(), { kernel_name });
    // synchronize
    builder_.CreateCall(llvm_decls_.get_nvvm_synchronize());

    // back-fetch to cpu
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        const Type* param_type = cuda_param->type();
        auto entry = device_ptrs[i - 4];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), entry.second };
        builder_.CreateCall(llvm_decls_.get_nvvm_read_memory(), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(llvm_decls_.get_nvvm_free_memory(), { device_ptr.first });
    return ret;
}

}
