#include "thorin/be/llvm/llvm.h"

namespace thorin {

Lam* CodeGen::emit_parallel(Lam* lam) {
    enum ParallelArgs {
        Mem, NumThreads, Lower, Upper, Body, Return, Last = Return
    };
    static constexpr size_t parallel_arg_count = size_t(ParallelArgs::Last) + 1;

    // arguments
    assert(lam->app()->num_args() >= parallel_arg_count && "required arguments are missing");
    auto num_threads = lookup(lam->app()->arg(ParallelArgs::NumThreads));
    auto lower = lookup(lam->app()->arg(ParallelArgs::Lower));
    auto upper = lookup(lam->app()->arg(ParallelArgs::Upper));
    auto kernel = lam->app()->arg(ParallelArgs::Body)->as<Global>()->init()->as_lam();

    const size_t num_kernel_args = lam->app()->num_args() - parallel_arg_count;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args + 1);
    par_args[0] = irbuilder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = lam->app()->arg(i + parallel_arg_count)->type();
        par_args[i + 1] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lam->app()->arg()->type()->as<TupleType>()->ops().skip_front(parallel_arg_count)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(lam->app()->arg(i + parallel_arg_count)), unsigned(i));
    } else {
        closure = lookup(lam->app()->arg(parallel_arg_count));
    }

    // allocate closure object and write values into it
    auto ptr = emit_alloca(closure_type, "parallel_closure");
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure, int lower, int upper)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_parallel_for";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft).getCallee()->stripPointerCasts();
    runtime_->parallel_for(num_threads, lower, upper, ptr, wrapper);

    // set insert point to the wrapper function
    auto old_bb = irbuilder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(*context_, wrapper_name, wrapper);
    irbuilder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = irbuilder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = irbuilder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args + 1);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            target_args[i + 1] = irbuilder_.CreateExtractValue(val, { unsigned(i) });
    } else {
        target_args[1] = val;
    }

    // create loop iterating over range:
    // for (int i=lower; i<upper; ++i)
    //   body(i, <closure_elems>);
    auto wrapper_lower = &*(++wrapper_args);
    auto wrapper_upper = &*(++wrapper_args);
    create_loop(wrapper_lower, wrapper_upper, irbuilder_.getInt32(1), wrapper, [&](llvm::Value* counter) {
        // call kernel body
        target_args[0] = counter; // loop index
        auto par_type = llvm::FunctionType::get(irbuilder_.getVoidTy(), llvm_ref(par_args), false);
        auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type).getCallee()->stripPointerCasts();
        irbuilder_.CreateCall(kernel_par_func, target_args);
    });
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    return lam->app()->arg(ParallelArgs::Return)->as_lam();
}

Lam* CodeGen::emit_fibers(Lam* lam) {
    enum FibersArgs {
        Mem, NumThreads, NumBlocks, NumWarps, Body, Return, Last = Return
    };
    static constexpr size_t fibers_arg_count = size_t(FibersArgs::Last) + 1;

    // arguments
    assert(lam->app()->num_args() >= fibers_arg_count && "required arguments are missing");
    auto num_threads = lookup(lam->app()->arg(FibersArgs::NumThreads));
    auto num_blocks = lookup(lam->app()->arg(FibersArgs::NumBlocks));
    auto num_warps = lookup(lam->app()->arg(FibersArgs::NumWarps));
    auto kernel = lam->app()->arg(FibersArgs::Body)->as<Global>()->init()->as_lam();

    const size_t num_kernel_args = lam->app()->num_args() - fibers_arg_count;

    // build fibers-function signature
    Array<llvm::Type*> fib_args(num_kernel_args + 2);
    fib_args[0] = irbuilder_.getInt32Ty(); // block index
    fib_args[1] = irbuilder_.getInt32Ty(); // warp index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = lam->app()->arg(i + fibers_arg_count)->type();
        fib_args[i + 2] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lam->app()->arg()->type()->as<TupleType>()->ops().skip_front(fibers_arg_count)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(lam->app()->arg(i + fibers_arg_count)), unsigned(i));
    } else {
        closure = lookup(lam->app()->arg(fibers_arg_count));
    }

    // allocate closure object and write values into it
    auto ptr = emit_alloca(closure_type, "fibers_closure");
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure, int lower, int upper)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_fibers";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft).getCallee()->stripPointerCasts();
    runtime_->spawn_fibers(num_threads, num_blocks, num_warps, ptr, wrapper);

    // set insert point to the wrapper function
    auto old_bb = irbuilder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(*context_, wrapper_name, wrapper);
    irbuilder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = irbuilder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = irbuilder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args + 2);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            target_args[i + 2] = irbuilder_.CreateExtractValue(val, { unsigned(i) });
    } else {
        target_args[2] = val;
    }

    auto wrapper_block = &*(++wrapper_args);
    auto wrapper_warp = &*(++wrapper_args);

    target_args[0] = wrapper_block;
    target_args[1] = wrapper_warp;

    // call kernel body
    auto fib_type = llvm::FunctionType::get(irbuilder_.getVoidTy(), llvm_ref(fib_args), false);
    auto kernel_fib_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), fib_type).getCallee()->stripPointerCasts();
    irbuilder_.CreateCall(kernel_fib_func, target_args);
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    return lam->app()->arg(FibersArgs::Return)->as_lam();
}

Lam* CodeGen::emit_spawn(Lam* lam) {
    enum SpawnArgs {
        Mem, Body, Return, Last = Return
    };
    static constexpr size_t spawn_arg_count = size_t(SpawnArgs::Last) + 1;

    assert(lam->app()->num_args() >= spawn_arg_count && "required arguments are missing");
    auto kernel = lam->app()->arg(SpawnArgs::Body)->as<Global>()->init()->as_lam();
    const size_t num_kernel_args = lam->app()->num_args() - spawn_arg_count;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = lam->app()->arg(i + spawn_arg_count)->type();
        par_args[i] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lam->app()->arg()->type()->as<TupleType>()->ops().skip_front(spawn_arg_count)));
    llvm::Value* closure = nullptr;
    if (closure_type->isStructTy()) {
        closure = llvm::UndefValue::get(closure_type);
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(lam->app()->arg(i + spawn_arg_count)), unsigned(i));
    } else {
        closure = lookup(lam->app()->arg(0 + spawn_arg_count));
    }

    // allocate closure object and write values into it
    auto ptr = irbuilder_.CreateAlloca(closure_type, nullptr);
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0) };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_spawn_thread";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft).getCallee()->stripPointerCasts();
    auto call = runtime_->spawn_thread(ptr, wrapper);

    // set insert point to the wrapper function
    auto old_bb = irbuilder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(*context_, wrapper_name, wrapper);
    irbuilder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = irbuilder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = irbuilder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args);
    if (val->getType()->isStructTy()) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            target_args[i] = irbuilder_.CreateExtractValue(val, { unsigned(i) });
    } else {
        target_args[0] = val;
    }

    // call kernel body
    auto par_type = llvm::FunctionType::get(irbuilder_.getVoidTy(), llvm_ref(par_args), false);
    auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type).getCallee()->stripPointerCasts();
    irbuilder_.CreateCall(kernel_par_func, target_args);
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    // bind parameter of lam to received handle
    auto l = lam->app()->arg(SpawnArgs::Return)->as_lam();
    emit_result_phi(l->param(1), call);
    return l;
}


Lam* CodeGen::emit_sync(Lam* lam) {
    enum SyncArgs {
        Mem, Id, Return, Last = Return
    };
    static constexpr size_t sync_arg_count = size_t(SyncArgs::Last) + 1;

    assert(lam->app()->num_args() == sync_arg_count && "wrong number of arguments");
    auto id = lookup(lam->app()->arg(SyncArgs::Id));
    runtime_->sync_thread(id);
    return lam->app()->arg(SyncArgs::Return)->as_lam();
}

}

