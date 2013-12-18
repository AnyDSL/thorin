#include "thorin/be/llvm/nvvm.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

NVVMCodeGen::NVVMCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::PTX_Device)
{
    module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");

}

llvm::Function* NVVMCodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module_);
    f->setCallingConv(llvm::CallingConv::PTX_Kernel);
    // append required metadata
    llvm::NamedMDNode* annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Value* annotation_values[] = {
        f,
        llvm::MDString::get(context_, lambda->name),
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
    auto kernel = lambda->arg(2)->as<Global>()->init()->as<Lambda>()->name;
    Lambda* ret = lambda->arg(3)->as_lambda();

    // load kernel
    llvm::Value* module_name = builder_.CreateGlobalStringPtr(world_.name() + "_nvvm.ll");
    llvm::Value* kernel_name = builder_.CreateGlobalStringPtr(kernel);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(nvvm_load_kernel_, load_args);
    // fetch values and create external calls for intialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = cuda_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        llvm::Constant* size = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), num_elems);
        auto alloca = builder_.CreateAlloca(nvvm_device_ptr_ty_);
        auto device_ptr = builder_.CreateCall(nvvm_malloc_memory_, size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::pair<llvm::Value*, llvm::Constant*>(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), size };
        builder_.CreateCall(nvvm_write_memory_, mem_args);
        builder_.CreateCall(nvvm_set_kernel_arg_, { alloca });
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), it_space_x),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), 1),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context_), 1)
    };
    builder_.CreateCall(nvvm_set_problem_size_, problem_size_args);
    // launch
    builder_.CreateCall(nvvm_launch_kernel_, { kernel_name });
    // synchronize
    builder_.CreateCall(nvvm_synchronize_);

    // back-fetch to cpu
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        auto entry = device_ptrs[i - 4];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), entry.second };
        builder_.CreateCall(nvvm_read_memory_, args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(nvvm_free_memory_, { device_ptr.first });
    return ret;
}

}
