#include "thorin/be/llvm/llvm.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

llvm::Function* CodeGen::emit_nnvm_function_decl(llvm::LLVMContext& context, llvm::Module* mod, std::string& name, Lambda* lambda) {
    llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "kernel", mod);
    f->setCallingConv(llvm::CallingConv::PTX_Kernel);
    // append required metadata
    llvm::NamedMDNode* annotation = mod->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Value* annotation_values[] = {
        f,
        llvm::MDString::get(context, lambda->unique_name()),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1)
    };
    annotation->addOperand(llvm::MDNode::get(context, annotation_values));
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
    llvm::Value* module_name = builder.CreateGlobalStringPtr(kernel->unique_name());
    llvm::Value* kernel_name = builder.CreateGlobalStringPtr("kernel");
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder.CreateCall(llvm_decls.get_nvvm_load_kernel(), load_args);
    // fetch values and create external calls for intialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = cuda_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        llvm::Constant* size = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), num_elems);
        auto alloca = builder.CreateAlloca(llvm_decls.get_nvvm_device_ptr_type());
        auto device_ptr = builder.CreateCall(llvm_decls.get_nvvm_malloc_memory(), size);
        // store device ptr
        builder.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder.CreateLoad(alloca);
        device_ptrs.push_back(std::pair<llvm::Value*, llvm::Constant*>(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, builder.CreateBitCast(lookup(cuda_param), builder.getInt8PtrTy()), size };
        builder.CreateCall(llvm_decls.get_nvvm_write_memory(), mem_args);
        builder.CreateCall(llvm_decls.get_nvvm_set_kernel_arg(), { alloca });
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), it_space_x),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1)
    };
    builder.CreateCall(llvm_decls.get_nvvm_set_problem_size(), problem_size_args);
    // launch
    builder.CreateCall(llvm_decls.get_nvvm_launch_kernel(), { kernel_name });
    // synchronize
    builder.CreateCall(llvm_decls.get_nvvm_synchronize());

    // back-fetch to cpu
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        const Type* param_type = cuda_param->type();
        auto entry = device_ptrs[i - 4];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, builder.CreateBitCast(lookup(cuda_param), builder.getInt8PtrTy()), entry.second };
        builder.CreateCall(llvm_decls.get_nvvm_read_memory(), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder.CreateCall(llvm_decls.get_nvvm_free_memory(), { device_ptr.first });
    return ret;
}

}
