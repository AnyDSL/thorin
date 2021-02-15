#include "thorin/be/llvm/runtime.h"

#include <sstream>
#include <stdexcept>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/be/llvm/runtime.inc"

namespace thorin::llvm {

Runtime::Runtime(llvm::LLVMContext& context,
                 llvm::Module& target)
    : target_(target)
    , layout_(target.getDataLayout())
{
    llvm::SMDiagnostic diag;
    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(runtime_definitions);
    runtime_ = llvm::parseIR(*mem_buf.get(), diag, context);
    if (runtime_ == nullptr)
        throw std::logic_error("runtime could not be loaded");
}

llvm::Function* Runtime::get(const char* name) {
    auto result = llvm::cast<llvm::Function>(target_.getOrInsertFunction(name, runtime_->getFunction(name)->getFunctionType()).getCallee()->stripPointerCasts());
    assert(result != nullptr && "Required runtime function could not be resolved");
    return result;
}

static bool contains_ptrtype(const Type* type) {
    switch (type->tag()) {
        case Node_PtrType:             return false;
        case Node_IndefiniteArrayType: return contains_ptrtype(type->as<ArrayType>()->elem_type());
        case Node_DefiniteArrayType:   return contains_ptrtype(type->as<DefiniteArrayType>()->elem_type());
        case Node_FnType:              return false;
        case Node_StructType: {
            bool good = true;
            auto struct_type = type->as<StructType>();
            for (size_t i = 0, e = struct_type->num_ops(); i != e; ++i)
                good &= contains_ptrtype(struct_type->op(i));
            return good;
        }
        case Node_TupleType: {
            bool good = true;
            auto tuple = type->as<TupleType>();
            for (size_t i = 0, e = tuple->num_ops(); i != e; ++i)
                good &= contains_ptrtype(tuple->op(i));
            return good;
        }
        default: return true;
    }
}

Continuation* Runtime::emit_host_code(CodeGen& code_gen, llvm::IRBuilder<>& builder, Platform platform, const std::string& ext, Continuation* continuation) {
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = continuation->callee()->as_continuation();
    assert_unused(target->is_intrinsic());
    assert(continuation->num_args() >= LaunchArgs::Num && "required arguments are missing");

    // arguments
    auto target_device_id = code_gen.emit(continuation->arg(LaunchArgs::Device));
    auto target_platform = builder.getInt32(platform);
    auto target_device = builder.CreateOr(target_platform, builder.CreateShl(target_device_id, builder.getInt32(4)));

    auto it_space = continuation->arg(LaunchArgs::Space);
    auto it_config = continuation->arg(LaunchArgs::Config);
    auto kernel = continuation->arg(LaunchArgs::Body)->as<Global>()->init()->as<Continuation>();

    auto& world = continuation->world();
    auto kernel_name = builder.CreateGlobalStringPtr(kernel->name());
    auto file_name = builder.CreateGlobalStringPtr(world.name() + ext);
    const size_t num_kernel_args = continuation->num_args() - LaunchArgs::Num;

    // allocate argument pointers, sizes, and types
    llvm::Value* args   = code_gen.emit_alloca(builder, llvm::ArrayType::get(builder.getInt8PtrTy(), num_kernel_args), "args");
    llvm::Value* sizes  = code_gen.emit_alloca(builder, llvm::ArrayType::get(builder.getInt32Ty(),   num_kernel_args), "sizes");
    llvm::Value* aligns = code_gen.emit_alloca(builder, llvm::ArrayType::get(builder.getInt32Ty(),   num_kernel_args), "aligns");
    llvm::Value* allocs = code_gen.emit_alloca(builder, llvm::ArrayType::get(builder.getInt32Ty(),   num_kernel_args), "allocs");
    llvm::Value* types  = code_gen.emit_alloca(builder, llvm::ArrayType::get(builder.getInt8Ty(),    num_kernel_args), "types");

    // fill array of arguments
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto target_arg = continuation->arg(i + LaunchArgs::Num);
        const auto target_val = code_gen.emit(target_arg);

        KernelArgType arg_type;
        llvm::Value*  void_ptr;
        if (target_arg->type()->isa<DefiniteArrayType>() ||
            target_arg->type()->isa<StructType>() ||
            target_arg->type()->isa<TupleType>()) {
            // definite array | struct | tuple
            auto alloca = code_gen.emit_alloca(builder, target_val->getType(), target_arg->name());
            builder.CreateStore(target_val, alloca);

            // check if argument type contains pointers
            if (!contains_ptrtype(target_arg->type()))
                world.wdef(target_arg, "argument '{}' of aggregate type '{}' contains pointer (not supported in OpenCL 1.2)", target_arg, target_arg->type());

            void_ptr = builder.CreatePointerCast(alloca, builder.getInt8PtrTy());
            arg_type = KernelArgType::Struct;
        } else if (target_arg->type()->isa<PtrType>()) {
            auto ptr = target_arg->type()->as<PtrType>();
            auto rtype = ptr->pointee();

            if (!rtype->isa<ArrayType>())
                world.edef(target_arg, "currently only pointers to arrays supported as kernel argument; argument has different type: {}", ptr);

            auto alloca = code_gen.emit_alloca(builder, builder.getInt8PtrTy(), target_arg->name());
            auto target_ptr = builder.CreatePointerCast(target_val, builder.getInt8PtrTy());
            builder.CreateStore(target_ptr, alloca);
            void_ptr = builder.CreatePointerCast(alloca, builder.getInt8PtrTy());
            arg_type = KernelArgType::Ptr;
        } else {
            // normal variable
            auto alloca = code_gen.emit_alloca(builder, target_val->getType(), target_arg->name());
            builder.CreateStore(target_val, alloca);

            void_ptr = builder.CreatePointerCast(alloca, builder.getInt8PtrTy());
            arg_type = KernelArgType::Val;
        }

        auto arg_ptr   = builder.CreateInBoundsGEP(args,   llvm::ArrayRef<llvm::Value*>{builder.getInt32(0), builder.getInt32(i)});
        auto size_ptr  = builder.CreateInBoundsGEP(sizes,  llvm::ArrayRef<llvm::Value*>{builder.getInt32(0), builder.getInt32(i)});
        auto align_ptr = builder.CreateInBoundsGEP(aligns, llvm::ArrayRef<llvm::Value*>{builder.getInt32(0), builder.getInt32(i)});
        auto alloc_ptr = builder.CreateInBoundsGEP(allocs, llvm::ArrayRef<llvm::Value*>{builder.getInt32(0), builder.getInt32(i)});
        auto type_ptr  = builder.CreateInBoundsGEP(types,  llvm::ArrayRef<llvm::Value*>{builder.getInt32(0), builder.getInt32(i)});

        auto size = layout_.getTypeStoreSize(target_val->getType()).getFixedSize();
        if (auto struct_type = llvm::dyn_cast<llvm::StructType>(target_val->getType())) {
            // In the case of a structure, do not include the padding at the end in the size
            auto last_elem   = struct_type->getStructNumElements() - 1;
            auto last_offset = layout_.getStructLayout(struct_type)->getElementOffset(last_elem);
            size = last_offset + layout_.getTypeStoreSize(struct_type->getStructElementType(last_elem)).getFixedSize();
        }

        builder.CreateStore(void_ptr, arg_ptr);
        builder.CreateStore(builder.getInt32(size), size_ptr);
        builder.CreateStore(builder.getInt32(layout_.getABITypeAlignment(target_val->getType())), align_ptr);
        builder.CreateStore(builder.getInt32(layout_.getTypeAllocSize(target_val->getType())), alloc_ptr);
        builder.CreateStore(builder.getInt8((uint8_t)arg_type), type_ptr);
    }

    // allocate arrays for the grid and block size
    const auto get_u32 = [&](const Def* def) { return builder.CreateSExt(code_gen.emit(def), builder.getInt32Ty()); };

    llvm::Value* grid_array  = llvm::UndefValue::get(llvm::ArrayType::get(builder.getInt32Ty(), 3));
    grid_array = builder.CreateInsertValue(grid_array, get_u32(world.extract(it_space, 0_u32)), 0);
    grid_array = builder.CreateInsertValue(grid_array, get_u32(world.extract(it_space, 1_u32)), 1);
    grid_array = builder.CreateInsertValue(grid_array, get_u32(world.extract(it_space, 2_u32)), 2);
    llvm::Value* grid_size = code_gen.emit_alloca(builder, grid_array->getType(), "");
    builder.CreateStore(grid_array, grid_size);

    llvm::Value* block_array = llvm::UndefValue::get(llvm::ArrayType::get(builder.getInt32Ty(), 3));
    block_array = builder.CreateInsertValue(block_array, get_u32(world.extract(it_config, 0_u32)), 0);
    block_array = builder.CreateInsertValue(block_array, get_u32(world.extract(it_config, 1_u32)), 1);
    block_array = builder.CreateInsertValue(block_array, get_u32(world.extract(it_config, 2_u32)), 2);
    llvm::Value* block_size = code_gen.emit_alloca(builder, block_array->getType(), "");
    builder.CreateStore(block_array, block_size);

    std::vector<llvm::Value*> gep_first_elem{builder.getInt32(0), builder.getInt32(0)};
    grid_size  = builder.CreateInBoundsGEP(grid_size,  gep_first_elem);
    block_size = builder.CreateInBoundsGEP(block_size, gep_first_elem);
    args       = builder.CreateInBoundsGEP(args,       gep_first_elem);
    sizes      = builder.CreateInBoundsGEP(sizes,      gep_first_elem);
    aligns     = builder.CreateInBoundsGEP(aligns,     gep_first_elem);
    allocs     = builder.CreateInBoundsGEP(allocs,     gep_first_elem);
    types      = builder.CreateInBoundsGEP(types,      gep_first_elem);

    launch_kernel(builder, target_device,
                  file_name, kernel_name,
                  grid_size, block_size,
                  args, sizes, aligns, allocs, types,
                  builder.getInt32(num_kernel_args));

    return continuation->arg(LaunchArgs::Return)->as_continuation();
}

llvm::Value* Runtime::launch_kernel(llvm::IRBuilder<>& builder, llvm::Value* device,
                                    llvm::Value* file, llvm::Value* kernel,
                                    llvm::Value* grid, llvm::Value* block,
                                    llvm::Value* args, llvm::Value* sizes, llvm::Value* aligns, llvm::Value* allocs, llvm::Value* types,
                                    llvm::Value* num_args) {
    llvm::Value* launch_args[] = { device, file, kernel, grid, block, args, sizes, aligns, allocs, types, num_args };
    return builder.CreateCall(get("anydsl_launch_kernel"), launch_args);
}

llvm::Value* Runtime::parallel_for(llvm::IRBuilder<>& builder, llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                                   llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* parallel_args[] = {
        num_threads, lower, upper,
        builder.CreatePointerCast(closure_ptr, builder.getInt8PtrTy()),
        builder.CreatePointerCast(fun_ptr, builder.getInt8PtrTy())
    };
    return builder.CreateCall(get("anydsl_parallel_for"), parallel_args);
}

llvm::Value* Runtime::spawn_fibers(llvm::IRBuilder<>& builder, llvm::Value* num_threads, llvm::Value* num_blocks, llvm::Value* num_warps,
                                   llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* fibers_args[] = {
        num_threads, num_blocks, num_warps,
        builder.CreatePointerCast(closure_ptr, builder.getInt8PtrTy()),
        builder.CreatePointerCast(fun_ptr, builder.getInt8PtrTy())
    };
    return builder.CreateCall(get("anydsl_fibers_spawn"), fibers_args);
}

llvm::Value* Runtime::spawn_thread(llvm::IRBuilder<>& builder, llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* spawn_args[] = {
        builder.CreatePointerCast(closure_ptr, builder.getInt8PtrTy()),
        builder.CreatePointerCast(fun_ptr, builder.getInt8PtrTy())
    };
    return builder.CreateCall(get("anydsl_spawn_thread"), spawn_args);
}

llvm::Value* Runtime::sync_thread(llvm::IRBuilder<>& builder, llvm::Value* id) {
    return builder.CreateCall(get("anydsl_sync_thread"), id);
}

}
