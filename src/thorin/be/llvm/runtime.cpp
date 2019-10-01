#include "thorin/be/llvm/runtime.h"

#include <sstream>
#include <stdexcept>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/be/llvm/llvm.h"
#include "thorin/be/llvm/runtime.inc"

namespace thorin {

Runtime::Runtime(llvm::LLVMContext& context,
                 llvm::Module& target,
                 llvm::IRBuilder<>& builder)
    : target_(target)
    , builder_(builder)
    , layout_(target.getDataLayout())
{
    llvm::SMDiagnostic diag;
    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(runtime_definitions);
    runtime_ = llvm::parseIR(*mem_buf.get(), diag, context);
    if (runtime_ == nullptr)
        throw std::logic_error("runtime could not be loaded");
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_.getOrInsertFunction(name, runtime_->getFunction(name)->getFunctionType()));
    assert(result != nullptr && "Required runtime function could not be resolved");
    return result;
}

static bool contains_ptrtype(const Def* type) {
    if (isa<Tag::Ptr>(type)) return false;
    switch (type->node()) {
        case Node::Variadic: return contains_ptrtype(type->as<Variadic>()->codomain());
        case Node::Pi:       return false;
        case Node::Sigma: {
            // TODO deal with recursive sigmas
            bool good = true;
            auto sigma = type->as<Sigma>();
            for (size_t i = 0, e = sigma->num_ops(); i != e; ++i)
                good &= contains_ptrtype(sigma->op(i));
            return good;
        }
        default: return true;
    }
}

Lam* Runtime::emit_host_code(CodeGen& code_gen, Platform platform, const std::string& ext, Lam* lam) {
    auto& world = lam->world();

    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lam->app()->callee()->as_nominal<Lam>();
    assert_unused(target->is_intrinsic());
    assert(lam->app()->num_args() >= LaunchArgs::Num && "required arguments are missing");

    // arguments
    auto target_device_id = code_gen.lookup(lam->app()->arg(LaunchArgs::Device));
    auto target_platform = builder_.getInt32(platform);
    auto target_device = builder_.CreateOr(target_platform, builder_.CreateShl(target_device_id, builder_.getInt32(4)));
    auto it_space = lam->app()->arg(LaunchArgs::Space)->as<Tuple>();
    auto it_config = lam->app()->arg(LaunchArgs::Config)->as<Tuple>();
    auto kernel = lam->app()->arg(LaunchArgs::Body)->as<Global>()->init()->as<Lam>();

    auto kernel_name = builder_.CreateGlobalStringPtr(kernel->name());
    auto file_name = builder_.CreateGlobalStringPtr(lam->world().name() + ext);
    const size_t num_kernel_args = lam->app()->num_args() - LaunchArgs::Num;

    // allocate argument pointers, sizes, and types
    llvm::Value* args   = code_gen.emit_alloca(llvm::ArrayType::get(builder_.getInt8PtrTy(), num_kernel_args), "args");
    llvm::Value* sizes  = code_gen.emit_alloca(llvm::ArrayType::get(builder_.getInt32Ty(),   num_kernel_args), "sizes");
    llvm::Value* aligns = code_gen.emit_alloca(llvm::ArrayType::get(builder_.getInt32Ty(),   num_kernel_args), "aligns");
    llvm::Value* types  = code_gen.emit_alloca(llvm::ArrayType::get(builder_.getInt8Ty(),    num_kernel_args), "types");

    // fill array of arguments
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto target_arg = lam->app()->arg(i + LaunchArgs::Num);
        const auto target_val = code_gen.lookup(target_arg);

        KernelArgType arg_type;
        llvm::Value*  void_ptr;
        if (target_arg->type()->isa<Variadic>() || target_arg->type()->isa<Sigma>()) {
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name());
            builder_.CreateStore(target_val, alloca);

            // check if argument type contains pointers
            if (!contains_ptrtype(target_arg->type()))
                world.wdef(target_arg, "argument '{}' of aggregate type '{}' contains pointer (not supported in OpenCL 1.2)", target_arg, target_arg->type());

            void_ptr = builder_.CreatePointerCast(alloca, builder_.getInt8PtrTy());
            arg_type = KernelArgType::Struct;
        } else if (auto ptr = isa<Tag::Ptr>(target_arg->type())) {
            auto [rtype, addr_space] = ptr->args<2>();

            if (!rtype->isa<Variadic>())
                world.edef(target_arg, "currently only pointers to arrays supported as kernel argument; argument has different type: {}", ptr);

            auto alloca = code_gen.emit_alloca(builder_.getInt8PtrTy(), target_arg->name());
            auto target_ptr = builder_.CreatePointerCast(target_val, builder_.getInt8PtrTy());
            builder_.CreateStore(target_ptr, alloca);
            void_ptr = builder_.CreatePointerCast(alloca, builder_.getInt8PtrTy());
            arg_type = KernelArgType::Ptr;
        } else {
            // normal variable
            auto alloca = code_gen.emit_alloca(target_val->getType(), target_arg->name());
            builder_.CreateStore(target_val, alloca);

            void_ptr = builder_.CreatePointerCast(alloca, builder_.getInt8PtrTy());
            arg_type = KernelArgType::Val;
        }

        auto arg_ptr   = builder_.CreateInBoundsGEP(args,   llvm::ArrayRef<llvm::Value*>{builder_.getInt32(0), builder_.getInt32(i)});
        auto size_ptr  = builder_.CreateInBoundsGEP(sizes,  llvm::ArrayRef<llvm::Value*>{builder_.getInt32(0), builder_.getInt32(i)});
        auto align_ptr = builder_.CreateInBoundsGEP(aligns, llvm::ArrayRef<llvm::Value*>{builder_.getInt32(0), builder_.getInt32(i)});
        auto type_ptr  = builder_.CreateInBoundsGEP(types,  llvm::ArrayRef<llvm::Value*>{builder_.getInt32(0), builder_.getInt32(i)});

        auto size = layout_.getTypeStoreSize(target_val->getType());
        if (auto struct_type = llvm::dyn_cast<llvm::StructType>(target_val->getType())) {
            // In the case of a structure, do not include the padding at the end in the size
            auto last_elem   = struct_type->getStructNumElements() - 1;
            auto last_offset = layout_.getStructLayout(struct_type)->getElementOffset(last_elem);
            size = last_offset + layout_.getTypeStoreSize(struct_type->getStructElementType(last_elem));
        }

        builder_.CreateStore(void_ptr, arg_ptr);
        builder_.CreateStore(builder_.getInt32(size), size_ptr);
        builder_.CreateStore(builder_.getInt32(layout_.getABITypeAlignment(target_val->getType())), align_ptr);
        builder_.CreateStore(builder_.getInt8((uint8_t)arg_type), type_ptr);
    }

    // allocate arrays for the grid and block size
    const auto get_u32 = [&](const Def* def) { return builder_.CreateSExt(code_gen.lookup(def), builder_.getInt32Ty()); };

    llvm::Value* grid_array  = llvm::UndefValue::get(llvm::ArrayType::get(builder_.getInt32Ty(), 3));
    grid_array = builder_.CreateInsertValue(grid_array, get_u32(it_space->op(0)), 0);
    grid_array = builder_.CreateInsertValue(grid_array, get_u32(it_space->op(1)), 1);
    grid_array = builder_.CreateInsertValue(grid_array, get_u32(it_space->op(2)), 2);
    llvm::Value* grid_size = code_gen.emit_alloca(grid_array->getType(), "");
    builder_.CreateStore(grid_array, grid_size);

    llvm::Value* block_array = llvm::UndefValue::get(llvm::ArrayType::get(builder_.getInt32Ty(), 3));
    block_array = builder_.CreateInsertValue(block_array, get_u32(it_config->op(0)), 0);
    block_array = builder_.CreateInsertValue(block_array, get_u32(it_config->op(1)), 1);
    block_array = builder_.CreateInsertValue(block_array, get_u32(it_config->op(2)), 2);
    llvm::Value* block_size = code_gen.emit_alloca(block_array->getType(), "");
    builder_.CreateStore(block_array, block_size);

    std::vector<llvm::Value*> gep_first_elem{builder_.getInt32(0), builder_.getInt32(0)};
    grid_size  = builder_.CreateInBoundsGEP(grid_size,  gep_first_elem);
    block_size = builder_.CreateInBoundsGEP(block_size, gep_first_elem);
    args       = builder_.CreateInBoundsGEP(args,       gep_first_elem);
    sizes      = builder_.CreateInBoundsGEP(sizes,      gep_first_elem);
    aligns     = builder_.CreateInBoundsGEP(aligns,     gep_first_elem);
    types      = builder_.CreateInBoundsGEP(types,      gep_first_elem);

    launch_kernel(target_device,
                  file_name, kernel_name,
                  grid_size, block_size,
                  args, sizes, aligns, types,
                  builder_.getInt32(num_kernel_args));

    return lam->app()->arg(LaunchArgs::Return)->as_nominal<Lam>();
}

llvm::Value* Runtime::launch_kernel(llvm::Value* device,
                                    llvm::Value* file, llvm::Value* kernel,
                                    llvm::Value* grid, llvm::Value* block,
                                    llvm::Value* args, llvm::Value* sizes, llvm::Value* aligns, llvm::Value* types,
                                    llvm::Value* num_args) {
    llvm::Value* launch_args[] = { device, file, kernel, grid, block, args, sizes, aligns, types, num_args };
    return builder_.CreateCall(get("anydsl_launch_kernel"), launch_args);
}

llvm::Value* Runtime::parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                                   llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* parallel_args[] = {
        num_threads, lower, upper,
        builder_.CreatePointerCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.CreatePointerCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("anydsl_parallel_for"), parallel_args);
}

llvm::Value* Runtime::spawn_thread(llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* spawn_args[] = {
        builder_.CreatePointerCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.CreatePointerCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("anydsl_spawn_thread"), spawn_args);
}

llvm::Value* Runtime::sync_thread(llvm::Value* id) {
    return builder_.CreateCall(get("anydsl_sync_thread"), id);
}

}
