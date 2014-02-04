#include "thorin/be/llvm/spir.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

SPIRCodeGen::SPIRCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::SPIR_FUNC)
{
    module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024");
    module_->setTargetTriple("spir64-unknown-unknown");
}

llvm::Function* SPIRCodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    // iterate over function type and set address space for SPIR
    llvm::FunctionType* fty = llvm::dyn_cast<llvm::FunctionType>(map(lambda->world().pi(lambda->pi()->elems())));
    llvm::SmallVector<llvm::Type*, 4> types;
    llvm::Type* rtype = fty->getReturnType();
    if (llvm::isa<llvm::PointerType>(rtype))
        rtype = llvm::dyn_cast<llvm::PointerType>(rtype)->getElementType()->getPointerTo(1);
    for (size_t i = 0; i < fty->getFunctionNumParams(); ++i) {
        llvm::Type* ty = fty->getFunctionParamType(i);
        if (llvm::isa<llvm::PointerType>(ty))
            types.push_back(llvm::dyn_cast<llvm::PointerType>(ty)->getElementType()->getPointerTo(1));
        else
            types.push_back(ty);
    }

    auto ft = llvm::FunctionType::get(rtype, types, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, module_);

    // FIXME: assume that kernels return void, other functions not
    if (!rtype->isVoidTy()) return f;

    // append required metadata
    llvm::NamedMDNode* annotation;
    llvm::Value* annotation_values_12[] = { builder_.getInt32(1), builder_.getInt32(2) };
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
        annotation_values_addr_space[index] = builder_.getInt32(addr_space);
        annotation_values_access_qual[index] = llvm::MDString::get(context_, "none");
        std::string type_string;
        llvm::raw_string_ostream type_os(type_string);
        type->print(type_os);
        type_os.flush();
        annotation_values_type[index] = llvm::MDString::get(context_, type_string);
        annotation_values_type_qual[index] = llvm::MDString::get(context_, "");
        annotation_values_name[index] = llvm::MDString::get(context_, lambda->param(index + 1)->name);
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
    annotation = module_->getOrInsertNamedMetadata("opencl.kernels");
    annotation->addOperand(llvm::MDNode::get(context_, annotation_values_kernel));
    // opencl.enable.FP_CONTRACT
    annotation = module_->getOrInsertNamedMetadata("opencl.enable.FP_CONTRACT");
    // opencl.spir.version
    annotation = module_->getOrInsertNamedMetadata("opencl.spir.version");
    annotation->addOperand(llvm::MDNode::get(context_, annotation_values_12));
    // opencl.ocl.version
    annotation = module_->getOrInsertNamedMetadata("opencl.ocl.version");
    annotation->addOperand(llvm::MDNode::get(context_, annotation_values_12));
    // opencl.used.extensions
    annotation = module_->getOrInsertNamedMetadata("opencl.used.extensions");
    // opencl.used.optional.core.features
    annotation = module_->getOrInsertNamedMetadata("opencl.used.optional.core.features");
    // opencl.compiler.options
    annotation = module_->getOrInsertNamedMetadata("opencl.compiler.options");
    f->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    return f;
}

Lambda* CodeGen::emit_spir(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::SPIR));
    assert(lambda->num_args() > 4 && "required arguments are missing");

    // get input
    auto it_space  = lambda->arg(1)->as<Tuple>();
    auto it_config = lambda->arg(2)->as<Tuple>();
    auto kernel = lambda->arg(3)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(4)->as_lambda();

    // load kernel
    auto module_name = builder_.CreateGlobalStringPtr(world_.name() + "_spir.bc");
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(spir("spir_build_program_and_kernel"), load_args);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def spir_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = spir_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto alloca = builder_.CreateAlloca(spir_device_ptr_ty_);
        auto device_ptr = builder_.CreateCall(spir("spir_malloc_buffer"), size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        llvm::Value* mem_args[] = {
            loaded_device_ptr,
            builder_.CreateBitCast(lookup(spir_param), llvm::Type::getInt8PtrTy(context_)),
            size
        };
        builder_.CreateCall(spir("spir_write_buffer"), mem_args);
        // set_kernel_arg(void *, size_t)
        auto *DL = new llvm::DataLayout(module_.get());
        auto size_of_arg = builder_.getInt64(DL->getTypeAllocSize(llvm::Type::getInt8PtrTy(context_)));
        llvm::Value* arg_args[] = { alloca, size_of_arg };
        builder_.CreateCall(spir("spir_set_kernel_arg"), arg_args);
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(spir("spir_set_problem_size"), problem_size_args);
    // setup configuration
    llvm::Value* config_args[] = {
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(spir("spir_set_config_size"), config_args);
    // launch
    builder_.CreateCall(spir("spir_launch_kernel"), { kernel_name });
    // synchronize
    builder_.CreateCall(spir("spir_synchronize"));

    // back-fetch to CPU
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def spir_param = lambda->arg(i);
        auto entry = device_ptrs[i - 5];
        // need to fetch back memory
        llvm::Value* args[] = {
            entry.first,
            builder_.CreateBitCast(lookup(spir_param), llvm::Type::getInt8PtrTy(context_)),
            entry.second
        };
        builder_.CreateCall(spir("spir_read_buffer"), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(spir("spir_free_buffer"), { device_ptr.first });
    return ret;
}

}
