#include "thorin/be/llvm/runtime.h"

#include <sstream>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

Runtime::Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder, const char* mod_name)
    : target_(target)
    , builder_(builder)
{
    llvm::SMDiagnostic diag;
    module_ = llvm::ParseIRFile(mod_name, diag, context);
    assert(module_ != nullptr && "Runtime could not be loaded");
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_->getOrInsertFunction(name, module_->getFunction(name)->getFunctionType()));
    assert(result != nullptr && "Required runtime function could not be resolved");
    return result;
}

KernelRuntime::KernelRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder,
                             llvm::Type* device_ptr_ty, const char* mod_name)
    : Runtime(context, target, builder, mod_name)
    , device_ptr_ty_(device_ptr_ty)
{}

Lambda* KernelRuntime::emit_host_code(CodeGen &code_gen, Lambda* lambda) {
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin());
    assert(lambda->num_args() > 5 && "required arguments are missing");

    // get input
    auto target_device = int(lambda->arg(1)->as<PrimLit>()->qu32_value().data());
    auto target_device_val = builder_.getInt32(target_device);
    auto it_space  = lambda->arg(2)->as<Tuple>();
    auto it_config = lambda->arg(3)->as<Tuple>();
    auto kernel = lambda->arg(4)->as<Global>()->init()->as<Lambda>();
    auto ret = lambda->arg(5)->as_lambda();

    // load kernel
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel->name);
    load_kernel(target_device_val, builder_.CreateGlobalStringPtr(get_module_name(lambda)), kernel_name);

    // fetch values and create external calls for initialization
    // check for source devices of all pointers
    DefMap<llvm::Value*> device_ptrs;
    DefMap<llvm::Value*> mapped_ptrs;
    for (size_t i = 6, e = lambda->num_args(); i < e; ++i) {
        Def target_arg = lambda->arg(i);
        const auto target_val = code_gen.lookup(target_arg);

        // check device target
        if (target_arg->type().isa<PtrType>()) {
            auto ptr = target_arg->type().as<PtrType>();
            if (ptr->device() == target_device) {
                // data is already on this device
                mapped_ptrs[target_arg] = target_val;
                if (ptr->addr_space() == AddressSpace::Texture) {
                    // skip memory and return continuation of given kernel
                    auto target_param = kernel->param(i - 6 + 1 + 1);
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
    const auto get_u64 = [&](Def def) { return builder_.CreateSExt(code_gen.lookup(def), builder_.getInt64Ty()); };

    // setup configuration and launch
    set_problem_size(target_device_val, get_u64(it_space->op(0)), get_u64(it_space->op(1)), get_u64(it_space->op(2)));
    set_config_size(target_device_val, get_u64(it_config->op(0)), get_u64(it_config->op(1)), get_u64(it_config->op(2)));
    launch_kernel(target_device_val, kernel_name);

    // synchronize
    synchronize(target_device_val);

    // emit copy-back operations
    for (auto entry : device_ptrs)
        read(target_device_val, entry.second, code_gen.lookup(entry.first));
    // emit unmap operations
    for (auto entry : mapped_ptrs)
        code_gen.runtime_->munmap(target_device, (uint32_t)entry.first->type().as<PtrType>()->addr_space(), entry.second);
    // emit free operations
    for (auto entry : device_ptrs)
        free(target_device_val, entry.second);

    return ret;
}

}
