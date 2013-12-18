#include "thorin/be/llvm/llvm.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

llvm::Function* CodeGen::emit_spir_function_decl(llvm::LLVMContext& context, llvm::Module* mod, std::string& name, Lambda* lambda) {
    llvm::Type* ty = map(lambda->world().pi(lambda->pi()->elems()));
    // iterate over function type and set address space for SPIR
    llvm::SmallVector<llvm::Type*, 4> types;
    for (size_t i = 0; i < ty->getFunctionNumParams(); ++i) {
        llvm::Type* fty = ty->getFunctionParamType(i);
        if (llvm::isa<llvm::PointerType>(fty))
            types.push_back(llvm::dyn_cast<llvm::PointerType>(fty)->getElementType()->getPointerTo(1));
        else
            types.push_back(fty);
    }
    llvm::FunctionType* ft = llvm::FunctionType::get(llvm::IntegerType::getVoidTy(context), types, false);
    llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "kernel", mod);
    f->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    // append required metadata
    llvm::NamedMDNode* annotation;
    llvm::Value* annotation_values_12[] = { llvm::ConstantInt::get(llvm::IntegerType::getInt32Ty(context), 1),
                                            llvm::ConstantInt::get(llvm::IntegerType::getInt32Ty(context), 2) };
    size_t num_params = lambda->num_params();
    Array<llvm::Value*> annotation_values_addr_space(num_params);
    Array<llvm::Value*> annotation_values_access_qual(num_params);
    Array<llvm::Value*> annotation_values_type(num_params);
    Array<llvm::Value*> annotation_values_type_qual(num_params);
    Array<llvm::Value*> annotation_values_name(num_params);
    annotation_values_addr_space[0]  = llvm::MDString::get(context, "kernel_arg_addr_space");
    annotation_values_access_qual[0] = llvm::MDString::get(context, "kernel_arg_access_qual");
    annotation_values_type[0]        = llvm::MDString::get(context, "kernel_arg_type");
    annotation_values_type_qual[0]   = llvm::MDString::get(context, "kernel_arg_type_qual");
    annotation_values_name[0]        = llvm::MDString::get(context, "kernel_arg_name");
    size_t index = 1;
    for (auto it = f->arg_begin(), e = f->arg_end(); it != e; ++it, ++index) {
        llvm::Type* type = it->getType();
        size_t addr_space = 0;
        if (llvm::isa<llvm::PointerType>(type)) {
            addr_space = llvm::dyn_cast<llvm::PointerType>(type)->getAddressSpace();
            type = llvm::dyn_cast<llvm::PointerType>(type)->getElementType()->getPointerTo(0);
        }
        annotation_values_addr_space[index] = llvm::ConstantInt::get(llvm::IntegerType::getInt32Ty(context), addr_space);
        annotation_values_access_qual[index] = llvm::MDString::get(context, "none");
        std::string type_string;
        llvm::raw_string_ostream type_os(type_string);
        type->print(type_os);
        type_os.flush();
        annotation_values_type[index] = llvm::MDString::get(context, type_string);
        annotation_values_type_qual[index] = llvm::MDString::get(context, "");
        annotation_values_name[index] = llvm::MDString::get(context, lambda->param(index - 1)->name);
    }
    llvm::Value* annotation_values_kernel[] = {
        f,
        llvm::MDNode::get(context, llvm_ref(annotation_values_addr_space)),
        llvm::MDNode::get(context, llvm_ref(annotation_values_access_qual)),
        llvm::MDNode::get(context, llvm_ref(annotation_values_type)),
        llvm::MDNode::get(context, llvm_ref(annotation_values_type_qual)),
        llvm::MDNode::get(context, llvm_ref(annotation_values_name))
    };
    // opencl.kernels
    annotation = mod->getOrInsertNamedMetadata("opencl.kernels");
    annotation->addOperand(llvm::MDNode::get(context, annotation_values_kernel));
    // opencl.enable.FP_CONTRACT
    annotation = mod->getOrInsertNamedMetadata("opencl.enable.FP_CONTRACT");
    // opencl.spir.version
    annotation = mod->getOrInsertNamedMetadata("opencl.spir.version");
    annotation->addOperand(llvm::MDNode::get(context, annotation_values_12));
    // opencl.ocl.version
    annotation = mod->getOrInsertNamedMetadata("opencl.ocl.version");
    annotation->addOperand(llvm::MDNode::get(context, annotation_values_12));
    // opencl.used.extensions
    annotation = mod->getOrInsertNamedMetadata("opencl.used.extensions");
    // opencl.used.optional.core.features
    annotation = mod->getOrInsertNamedMetadata("opencl.used.optional.core.features");
    // opencl.compiler.options
    annotation = mod->getOrInsertNamedMetadata("opencl.compiler.options");
    return f;
}

Lambda* CodeGen::emit_spir(Lambda* lambda) {
    Lambda* target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::SPIR));
    assert(lambda->num_args() > 3 && "required arguments are missing");
    // get input
    const uint64_t it_space_x = lambda->arg(1)->as<PrimLit>()->u64_value();
    Lambda* kernel = lambda->arg(2)->as_lambda();
    Lambda* ret = lambda->arg(3)->as_lambda();
    // load kernel
    llvm::Value* module_name = builder.CreateGlobalStringPtr(kernel->unique_name());
    llvm::Value* kernel_name = builder.CreateGlobalStringPtr("kernel");
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder.CreateCall(llvm_decls.get_spir_build_program_and_kernel(), load_args);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def spir_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = spir_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        llvm::Constant* size = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), num_elems);
        auto alloca = builder.CreateAlloca(llvm_decls.get_spir_device_ptr_type());
        auto device_ptr = builder.CreateCall(llvm_decls.get_spir_malloc_buffer(), size);
        // store device ptr
        builder.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder.CreateLoad(alloca);
        device_ptrs.push_back(std::pair<llvm::Value*, llvm::Constant*>(loaded_device_ptr, size));
        llvm::Value* mem_args[] = {
            loaded_device_ptr,
            builder.CreateBitCast(lookup(spir_param), llvm::Type::getInt8PtrTy(context)),
            size
        };
        builder.CreateCall(llvm_decls.get_spir_write_buffer(), mem_args);
        // set_kernel_arg(void *, size_t)
        const llvm::DataLayout *DL = new llvm::DataLayout(module.get());
        llvm::Value* size_of_arg = builder.getInt64(DL->getTypeAllocSize(llvm::Type::getInt8PtrTy(context)));
        llvm::Value* arg_args[] = { alloca, size_of_arg };
        builder.CreateCall(llvm_decls.get_spir_set_kernel_arg(), arg_args);
    }
    // determine problem size
    llvm::Value* problem_size_args[] = {
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), it_space_x),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1),
        llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(context), 1)
    };
    builder.CreateCall(llvm_decls.get_spir_set_problem_size(), problem_size_args);
    // launch
    builder.CreateCall(llvm_decls.get_spir_launch_kernel(), { kernel_name });
    // synchronize
    builder.CreateCall(llvm_decls.get_spir_synchronize());

    // fetch data back to CPU
    for (size_t i = 4, e = lambda->num_args(); i < e; ++i) {
        Def spir_param = lambda->arg(i);
        const Type* param_type = spir_param->type();
        auto entry = device_ptrs[i - 4];
        // need to fetch back memory
        llvm::Value* args[] = {
            entry.first,
            builder.CreateBitCast(lookup(spir_param), llvm::Type::getInt8PtrTy(context)),
            entry.second
        };
        builder.CreateCall(llvm_decls.get_spir_read_buffer(), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder.CreateCall(llvm_decls.get_spir_free_buffer(), { device_ptr.first });
    return ret;
}

}
