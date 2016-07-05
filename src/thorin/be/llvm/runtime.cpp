#include "thorin/be/llvm/runtime.h"

#include <sstream>
#include <stdexcept>

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/primop.h"
#include "thorin/util/log.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

Runtime::Runtime(llvm::LLVMContext& context,
                 llvm::Module* target,
                 llvm::IRBuilder<>& builder)
    : target_(target)
    , builder_(builder)
    , layout_(new llvm::DataLayout(target_))
{
    llvm::SMDiagnostic diag;
    runtime_ = llvm::ParseIRFile(THORIN_RUNTIME_PLATFORMS "runtime.s", diag, context);
    if (runtime_ == nullptr)
        throw std::logic_error("runtime could not be loaded");
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_->getOrInsertFunction(name, runtime_->getFunction(name)->getFunctionType()));
    assert(result != nullptr && "Required runtime function could not be resolved");
    return result;
}

enum {
    ACC_ARG_MEM,
    ACC_ARG_DEVICE,
    ACC_ARG_SPACE,
    ACC_ARG_CONFIG,
    ACC_ARG_BODY,
    ACC_ARG_RETURN,
    ACC_NUM_ARGS
};

Continuation* Runtime::emit_host_code(CodeGen& code_gen, Platform platform, const std::string& ext, Continuation* continuation) {
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = continuation->callee()->as_continuation();
    assert_unused(target->is_intrinsic());
    assert(continuation->num_args() >= ACC_NUM_ARGS && "required arguments are missing");

    // arguments
    auto target_device_id = code_gen.lookup(continuation->arg(ACC_ARG_DEVICE));
    auto target_platform = builder_.getInt32(platform);
    auto target_device = builder_.CreateOr(target_platform, builder_.CreateShl(target_device_id, builder_.getInt32(4)));
    auto it_space = continuation->arg(ACC_ARG_SPACE)->as<Tuple>();
    auto it_config = continuation->arg(ACC_ARG_CONFIG)->as<Tuple>();
    auto kernel = continuation->arg(ACC_ARG_BODY)->as<Global>()->init()->as<Continuation>();

    // load kernel
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel->name);
    auto file_name = builder_.CreateGlobalStringPtr(continuation->world().name() + ext);
    load_kernel(target_device, file_name, kernel_name);

    // fetch values and create external calls for initialization
    // check for source devices of all pointers
    const size_t num_kernel_args = continuation->num_args() - ACC_NUM_ARGS;
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto target_arg = continuation->arg(i + ACC_NUM_ARGS);
        const auto target_val = code_gen.lookup(target_arg);

        // check device target
        if (target_arg->type()->isa<DefiniteArrayType>() ||
            target_arg->type()->isa<StructAppType>() ||
            target_arg->type()->isa<TupleType>()) {
            // definite array | struct | tuple
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name);
            builder_.CreateStore(target_val, alloca);
            auto void_ptr = builder_.CreatePointerCast(alloca, builder_.getInt8PtrTy());
            // TODO: recurse over struct|tuple and check if it contains pointers
            set_kernel_arg_struct(target_device, i, void_ptr, target_val->getType());
        } else if (target_arg->type()->isa<PtrType>()) {
            auto ptr = target_arg->type()->as<PtrType>();
            auto rtype = ptr->referenced_type();

            if (!rtype->isa<ArrayType>())
                ELOG("currently only pointers to arrays supported as kernel argument at '%'; argument has different type: %", target_arg->loc(), ptr);

            auto void_ptr = builder_.CreatePointerCast(target_val, builder_.getInt8PtrTy());
            set_kernel_arg_ptr(target_device, i, void_ptr);
        } else {
            // normal variable
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name);
            builder_.CreateStore(target_val, alloca);
            auto void_ptr = builder_.CreatePointerCast(alloca, builder_.getInt8PtrTy());
            set_kernel_arg(target_device, i, void_ptr, target_val->getType());
        }
    }

    // setup configuration and launch
    const auto get_u32 = [&](const Def* def) { return builder_.CreateSExt(code_gen.lookup(def), builder_.getInt32Ty()); };
    set_grid_size(target_device, get_u32(it_space->op(0)), get_u32(it_space->op(1)), get_u32(it_space->op(2)));
    set_block_size(target_device, get_u32(it_config->op(0)), get_u32(it_config->op(1)), get_u32(it_config->op(2)));
    launch_kernel(target_device);

    // synchronize
    synchronize(target_device);

    return continuation->arg(ACC_ARG_RETURN)->as_continuation();
}

llvm::Value* Runtime::set_grid_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* grid_size_args[] = { device, x, y, z };
    return builder_.CreateCall(get("thorin_set_grid_size"), grid_size_args);
}

llvm::Value* Runtime::set_block_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* block_size_args[] = { device, x, y, z };
    return builder_.CreateCall(get("thorin_set_block_size"), block_size_args);
}

llvm::Value* Runtime::synchronize(llvm::Value* device) {
    llvm::Value* sync_args[] = { device };
    return builder_.CreateCall(get("thorin_synchronize"), sync_args);
}

llvm::Value* Runtime::set_kernel_arg(llvm::Value* device, int arg, llvm::Value* mem, llvm::Type* type) {
    llvm::Value* kernel_args[] = { device, builder_.getInt32(arg), mem, builder_.getInt32(layout_->getTypeAllocSize(type)) };
    return builder_.CreateCall(get("thorin_set_kernel_arg"), kernel_args);
}

llvm::Value* Runtime::set_kernel_arg_ptr(llvm::Value* device, int arg, llvm::Value* ptr) {
    llvm::Value* kernel_args[] = { device, builder_.getInt32(arg), ptr };
    return builder_.CreateCall(get("thorin_set_kernel_arg_ptr"), kernel_args);
}

llvm::Value* Runtime::set_kernel_arg_struct(llvm::Value* device, int arg, llvm::Value* mem, llvm::Type* type) {
    llvm::Value* kernel_args[] = { device, builder_.getInt32(arg), mem, builder_.getInt32(layout_->getTypeAllocSize(type)) };
    return builder_.CreateCall(get("thorin_set_kernel_arg_struct"), kernel_args);
}

llvm::Value* Runtime::load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) {
    llvm::Value* load_args[] = { device, file, kernel };
    return builder_.CreateCall(get("thorin_load_kernel"), load_args);
}

llvm::Value* Runtime::launch_kernel(llvm::Value* device) {
    llvm::Value* launch_args[] = { device };
    return builder_.CreateCall(get("thorin_launch_kernel"), launch_args);
}

llvm::Value* Runtime::parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                                   llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* parallel_args[] = {
        num_threads, lower, upper,
        builder_.CreatePointerCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.CreatePointerCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("thorin_parallel_for"), parallel_args);
}

llvm::Value* Runtime::spawn_thread(llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* spawn_args[] = {
        builder_.CreatePointerCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.CreatePointerCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("thorin_spawn_thread"), spawn_args);
}

llvm::Value* Runtime::sync_thread(llvm::Value* id) {
    return builder_.CreateCall(get("thorin_sync_thread"), id);
}

}
