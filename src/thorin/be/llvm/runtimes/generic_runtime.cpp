#include <llvm/IR/DataLayout.h>

#include "thorin/be/llvm/runtimes/generic_runtime.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

GenericRuntime::GenericRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : Runtime(context, target, builder, "generic.s")
    , context_(context)
{}

llvm::Value* GenericRuntime::mmap(uint32_t device, uint32_t addr_space, llvm::Value* ptr,
                                 llvm::Value* top_left, llvm::Value* region_size) {
    llvm::Value* mmap_args[] = {
        builder_.getInt32(device),
        builder_.getInt32(addr_space),
        builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()),
        builder_.CreateExtractValue(top_left, 0), // x
        builder_.CreateExtractValue(top_left, 1), // y
        builder_.CreateExtractValue(top_left, 2), // z
        builder_.CreateExtractValue(region_size, 0), // x
        builder_.CreateExtractValue(region_size, 1), // y
        builder_.CreateExtractValue(region_size, 2), // z
    };
    return builder_.CreateCall(get("map_memory"), mmap_args);
}

llvm::Value* GenericRuntime::munmap(uint32_t device, uint32_t addr_space, llvm::Value* mem) {
    llvm::Value* mmap_args[] = {
        builder_.getInt32(device),
        builder_.getInt32(addr_space),
        mem,
    };
    return builder_.CreateCall(get("unmap_memory"), mmap_args);
}

llvm::Value* GenericRuntime::parallel_create(llvm::Value* num_threads, llvm::Value* closure_ptr,
                                             uint64_t closure_size, llvm::Value* fun_ptr) {

    llvm::Value* parallel_args[] = {
        num_threads,
        builder_.CreateBitCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.getInt64(closure_size),
        builder_.CreateBitCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("parallel_create"), parallel_args);
}

llvm::Value* GenericRuntime::parallel_join(llvm::Value* handle) {
    return builder_.CreateCall(get("parallel_join"), { handle });
}

Lambda* GenericRuntime::emit_parallel_start_code(CodeGen& code_gen, Lambda* lambda) {
    auto &world = lambda->world();
    // to-target is the desired parallel-target call
    // target(mem, num_threads, body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::Parallel));
    assert(lambda->num_args() > 3 && "required arguments are missing");

    // get input
    auto num_threads  = code_gen.lookup(lambda->arg(1));
    auto func = lambda->arg(2)->as<Global>()->init()->as_lambda();
    auto ret = lambda->arg(3)->as_lambda();
    const auto arg_index = 4;

    // call parallel runtime function
    auto target_fun = code_gen.fcts_[func];
    llvm::Value* handle;
    if (lambda->num_args() > arg_index) {
        // fetch values and create a unified struct which contains all values (closure)
        auto closure_type = code_gen.convert(world.tuple_type(lambda->arg_fn_type()->elems().slice_from_begin(4)));
        llvm::Value* closure = llvm::UndefValue::get(closure_type);
        for (size_t i = arg_index, e = lambda->num_args(); i != e; ++i)
            closure = builder_.CreateInsertValue(closure, code_gen.lookup(lambda->arg(i)), unsigned(i - arg_index));

        // allocate closure object and write values into it
        AutoPtr<llvm::DataLayout> dl(new llvm::DataLayout(module_));
        auto ptr = builder_.CreateAlloca(closure_type, nullptr);
        builder_.CreateStore(closure, ptr, false);

        // create wrapper function that extracts all arguments from the closure
        auto ft = llvm::FunctionType::get(builder_.getVoidTy(), { builder_.getInt8PtrTy(0) }, false);
        auto wrapper_name = lambda->unique_name() + "_parallel";
        auto wrapper = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, wrapper_name, target_);

        handle = parallel_create(num_threads, ptr, dl->getTypeAllocSize(closure_type), wrapper);

        auto oldBB = builder_.GetInsertBlock();

        // emit wrapper function
        auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
        builder_.SetInsertPoint(bb);

        // load value in different thread and extract data from the closure
        auto load_ptr =  builder_.CreateBitCast(&*wrapper->arg_begin(), llvm::PointerType::get(closure_type, 0));
        auto val = builder_.CreateLoad(load_ptr);

        std::vector<llvm::Value*> target_args;
        for (size_t i = 0, e = lambda->num_args() - arg_index; i != e; ++i)
            target_args.push_back(builder_.CreateExtractValue(val, { unsigned(i) }));
        builder_.CreateCall(target_fun, target_args);
        builder_.CreateRetVoid();

        // restore old insert point
        builder_.SetInsertPoint(oldBB);
    } else {
        // no closure required
        handle = parallel_create(num_threads, llvm::ConstantPointerNull::get(builder_.getInt8PtrTy()), 0, target_fun);
    }

    // bind parameter of continuation to received handle
    if (ret->num_params() == 1)
        code_gen.params_[ret->param(0)] = handle;

    return ret;
}

Lambda* GenericRuntime::emit_parallel_join_code(CodeGen& code_gen, Lambda* lambda) {
    assert(false && "TODO");
    return lambda;
}

}

