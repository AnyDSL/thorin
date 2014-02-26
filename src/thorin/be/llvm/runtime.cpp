#include "thorin/be/llvm/runtime.h"

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include "llvm/Support/raw_ostream.h"
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
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_->getOrInsertFunction(name, module_->getFunction(name)->getFunctionType()));
    assert(result != nullptr && "Requires runtime function could not be resolved");
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
    auto target_device  = lambda->arg(1)->as<PrimLit>()->qu32_value().data();
    auto target_device_val = builder_.getInt32(target_device);
    auto it_space  = lambda->arg(2)->as<Tuple>();
    auto it_config = lambda->arg(3)->as<Tuple>();
    auto kernel = lambda->arg(4)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(5)->as_lambda();

    // load kernel
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    load_kernel(target_device_val, builder_.CreateGlobalStringPtr(get_module_name(lambda)), kernel_name);

    // fetch values and create external calls for initialization
    // check for source devices of all pointers
    DefMap<llvm::Value*> device_ptrs;
    for (size_t i = 6, e = lambda->num_args(); i < e; ++i) {
        Def target_arg = lambda->arg(i);
        const auto target_val = code_gen.lookup(target_arg);
        // check device target
        if (auto ptr = target_arg->type()->as<Ptr>()) {
            if (ptr->device() == target_device) {
                // data is already on this device
                set_mapped_kernel_arg(target_device_val, target_val);
            } else {
                // we need to allocate memory for this chunk on the target device
                auto ptr = malloc(target_device_val, target_val);
                device_ptrs[target_arg] = ptr;
                // copy memory to target device
                write(target_device_val, ptr, target_val);
                set_kernel_arg(target_device_val, ptr);
            }
        }
    }
    const auto get_u64 = [&](Def def) { return builder_.getInt64(def->as<PrimLit>()->qu64_value()); };

    // setup configuration and launch
    set_problem_size(target_device_val, get_u64(it_space->op(0)), get_u64(it_space->op(1)), get_u64(it_space->op(2)));
    set_config_size(target_device_val, get_u64(it_config->op(0)), get_u64(it_config->op(1)), get_u64(it_config->op(2)));
    launch_kernel(target_device_val, kernel_name);

    // synchronize
    synchronize(target_device_val);

    // emit copy-back operations
    for (auto entry : device_ptrs)
        read(target_device_val, entry.second, code_gen.lookup(entry.first));
    // emit free operations
    for (auto entry : device_ptrs)
        free(target_device_val, entry.second);

    return ret;
}

}
