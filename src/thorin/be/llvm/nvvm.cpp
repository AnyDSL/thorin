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
    auto ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module_);

    // FIXME: assume that kernels return void, other functions not
    if (!ft->getReturnType()->isVoidTy()) return f;

    // append required metadata
    auto annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Value* annotation_values[] = { f, llvm::MDString::get(context_, "kernel"), builder_.getInt64(1) };
    annotation->addOperand(llvm::MDNode::get(context_, annotation_values));
    f->setCallingConv(llvm::CallingConv::PTX_Kernel);
    return f;
}

llvm::Function* NVVMCodeGen::emit_intrinsic_decl(std::string& name, Lambda* lambda) {
    auto f = CodeGen::emit_function_decl(name, lambda);
    f->setAttributes(llvm::AttributeSet());
    return f;
}

Lambda* CodeGen::emit_nvvm(Lambda* lambda) {
    // to-target is the desired cuda call
    // target(mem, size, body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::NVVM));
    assert(lambda->num_args() > 4 && "required arguments are missing");

    // get input
    auto it_space  = lambda->arg(1)->as<Tuple>();
    auto it_config = lambda->arg(2)->as<Tuple>();
    auto kernel = lambda->arg(3)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(4)->as_lambda();

    // load kernel
    auto module_name = builder_.CreateGlobalStringPtr(world_.name() + "_nvvm.ll");
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(nvvm("nvvm_load_kernel"), load_args);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = cuda_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto alloca = builder_.CreateAlloca(nvvm_device_ptr_ty_);
        auto device_ptr = builder_.CreateCall(nvvm("nvvm_malloc_memory"), size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), size };
        builder_.CreateCall(nvvm("nvvm_write_memory"), mem_args);
        builder_.CreateCall(nvvm("nvvm_set_kernel_arg"), { alloca });
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(nvvm("nvvm_set_problem_size"), problem_size_args);
    // setup configuration
    llvm::Value* config_args[] = {
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(nvvm("nvvm_set_config_size"), config_args);
    // launch
    builder_.CreateCall(nvvm("nvvm_launch_kernel"), { kernel_name });
    // synchronize
    builder_.CreateCall(nvvm("nvvm_synchronize"));

    // back-fetch to CPU
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        auto entry = device_ptrs[i - 5];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), entry.second };
        builder_.CreateCall(nvvm("nvvm_read_memory"), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(nvvm("nvvm_free_memory"), { device_ptr.first });
    return ret;
}

}
