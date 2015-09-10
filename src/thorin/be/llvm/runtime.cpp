#include "thorin/be/llvm/runtime.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

Runtime::Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder, const char* mod_name)
    : target_(target)
    , builder_(builder)
{
    llvm::SMDiagnostic diag;
    runtime_ = llvm::ParseIRFile(mod_name, diag, context);
    if (runtime_ == nullptr)
        throw std::logic_error("runtime could not be loaded");
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_->getOrInsertFunction(name, runtime_->getFunction(name)->getFunctionType()));
    assert(result != nullptr && "Required runtime function could not be resolved");
    return result;
}

KernelRuntime::KernelRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder,
                             llvm::Type* device_ptr_ty, const char* mod_name)
    : Runtime(context, target, builder, mod_name)
    , device_ptr_ty_(device_ptr_ty)
{}

enum {
    ACC_ARG_MEM,
    ACC_ARG_DEVICE,
    ACC_ARG_SPACE,
    ACC_ARG_CONFIG,
    ACC_ARG_BODY,
    ACC_ARG_RETURN,
    ACC_NUM_ARGS
};

Lambda* KernelRuntime::emit_host_code(CodeGen &code_gen, Lambda* lambda) {
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_intrinsic());
    assert(lambda->num_args() >= ACC_NUM_ARGS && "required arguments are missing");

    // arguments
    assert(lambda->arg(ACC_ARG_DEVICE)->isa<PrimLit>() && "target device must be hard-coded");
    auto target_device = int(lambda->arg(ACC_ARG_DEVICE)->as<PrimLit>()->qu32_value().data());
    auto target_device_val = builder_.getInt32(target_device);
    auto it_space  = lambda->arg(ACC_ARG_SPACE)->as<Tuple>();
    auto it_config = lambda->arg(ACC_ARG_CONFIG)->as<Tuple>();
    auto kernel = lambda->arg(ACC_ARG_BODY)->as<Global>()->init()->as<Lambda>();

    // load kernel
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel->name);
    load_kernel(target_device_val, builder_.CreateGlobalStringPtr(get_module_name(lambda)), kernel_name);

    // fetch values and create external calls for initialization
    // check for source devices of all pointers
    DefMap<llvm::Value*> device_ptrs;
    const size_t num_kernel_args = lambda->num_args() - ACC_NUM_ARGS;
    for (size_t i = 0; i < num_kernel_args; ++i) {
        Def target_arg = lambda->arg(i + ACC_NUM_ARGS);
        const auto target_val = code_gen.lookup(target_arg);

        // check device target
        if (target_arg->type().isa<DefiniteArrayType>() ||
            target_arg->type().isa<StructAppType>() ||
            target_arg->type().isa<TupleType>()) {
            // definite array | struct | tuple
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name);
            builder_.CreateStore(target_val, alloca);
            auto void_ptr = builder_.CreateBitCast(alloca, builder_.getInt8PtrTy());
            // TODO: recurse over struct|tuple and check if it contains pointers
            set_kernel_arg_struct(target_device_val, void_ptr, target_val->getType());
        } else if (target_arg->type().isa<PtrType>()) {
            auto ptr = target_arg->type().as<PtrType>();
            auto rtype = ptr->referenced_type();

            if (!rtype.isa<ArrayType>()) {
                std::cout << "only pointers to arrays supported as kernel argument; got other pointer:" << std::endl;
                ptr->dump();
                assert(rtype.isa<ArrayType>() && "currently only pointers to arrays supported");
            }

            if (ptr->device() == target_device) {
                // data is already on this device
                if (ptr->addr_space() == AddressSpace::Texture) {
                    // skip memory and return continuation of given kernel
                    auto target_param = kernel->param(i + 1 + 1);
                    auto target_array_type = target_param->type().as<PtrType>()->referenced_type().as<ArrayType>();
                    auto texture_type = target_array_type->elem_type().as<PrimType>();
                    auto texture_name = builder_.CreateGlobalStringPtr(target_param->unique_name());
                    set_texture(target_device_val, target_val, texture_name, texture_type->primtype_kind());
                } else {
                    // bind mapped buffer
                    set_kernel_arg_map(target_device_val, target_val);
                }
            } else {
                // we need to allocate memory for this chunk on the target device
                auto mem = malloc(target_device_val, target_val);
                device_ptrs[target_arg] = mem;
                // copy memory to target device
                write(target_device_val, mem, target_val);
                set_kernel_arg_map(target_device_val, mem);
            }
        } else {
            // normal variable
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name);
            builder_.CreateStore(target_val, alloca);
            auto void_ptr = builder_.CreateBitCast(alloca, builder_.getInt8PtrTy());
            set_kernel_arg(target_device_val, void_ptr, target_val->getType());
        }
    }

    // setup configuration and launch
    const auto get_u32 = [&](Def def) { return builder_.CreateSExt(code_gen.lookup(def), builder_.getInt32Ty()); };
    set_problem_size(target_device_val, get_u32(it_space->op(0)), get_u32(it_space->op(1)), get_u32(it_space->op(2)));
    set_config_size(target_device_val, get_u32(it_config->op(0)), get_u32(it_config->op(1)), get_u32(it_config->op(2)));
    launch_kernel(target_device_val, kernel_name);

    // synchronize
    synchronize(target_device_val);

    // emit copy-back operations
    for (auto entry : device_ptrs)
        read(target_device_val, entry.second, code_gen.lookup(entry.first));
    // emit free operations
    for (auto entry : device_ptrs)
        free(target_device_val, entry.second);

    return lambda->arg(ACC_ARG_RETURN)->as_lambda();
}

}
