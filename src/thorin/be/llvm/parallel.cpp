#include "thorin/be/llvm/llvm.h"

namespace thorin {

enum {
    PAR_ARG_MEM,
    PAR_ARG_NUMTHREADS,
    PAR_ARG_LOWER,
    PAR_ARG_UPPER,
    PAR_ARG_BODY,
    PAR_ARG_RETURN,
    PAR_NUM_ARGS
};

Continuation* CodeGen::emit_parallel(Continuation* continuation) {
    // arguments
    assert(continuation->num_args() >= PAR_NUM_ARGS && "required arguments are missing");
    auto num_threads = lookup(continuation->arg(PAR_ARG_NUMTHREADS));
    auto lower = lookup(continuation->arg(PAR_ARG_LOWER));
    auto upper = lookup(continuation->arg(PAR_ARG_UPPER));
    auto kernel = continuation->arg(PAR_ARG_BODY)->as<Global>()->init()->as_continuation();

    const size_t num_kernel_args = continuation->num_args() - PAR_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args + 1);
    par_args[0] = irbuilder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = continuation->arg(i + PAR_NUM_ARGS)->type();
        par_args[i + 1] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(continuation->arg_fn_type()->ops().skip_front(PAR_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(continuation->arg(i + PAR_NUM_ARGS)), unsigned(i));
    } else {
        closure = lookup(continuation->arg(PAR_NUM_ARGS));
    }

    // allocate closure object and write values into it
    auto ptr = emit_alloca(closure_type, "parallel_closure");
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure, int lower, int upper)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_parallel_for";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
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
        auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
        irbuilder_.CreateCall(kernel_par_func, target_args);
    });
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    return continuation->arg(PAR_ARG_RETURN)->as_continuation();
}

enum {
    FIB_ARG_MEM,
    FIB_ARG_NUMTHREADS,
    FIB_ARG_NUMBLOCKS,
    FIB_ARG_NUMWARPS,
    FIB_ARG_BODY,
    FIB_ARG_RETURN,
    FIB_NUM_ARGS
};

Continuation* CodeGen::emit_fibers(Continuation* continuation) {
    // arguments
    assert(continuation->num_args() >= FIB_NUM_ARGS && "required arguments are missing");
    auto num_threads = lookup(continuation->arg(FIB_ARG_NUMTHREADS));
    auto num_blocks = lookup(continuation->arg(FIB_ARG_NUMBLOCKS));
    auto num_warps = lookup(continuation->arg(FIB_ARG_NUMWARPS));
    auto kernel = continuation->arg(FIB_ARG_BODY)->as<Global>()->init()->as_continuation();

    const size_t num_kernel_args = continuation->num_args() - FIB_NUM_ARGS;

    // build fibers-function signature
    Array<llvm::Type*> fib_args(num_kernel_args + 2);
    fib_args[0] = irbuilder_.getInt32Ty(); // block index
    fib_args[1] = irbuilder_.getInt32Ty(); // warp index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = continuation->arg(i + FIB_NUM_ARGS)->type();
        fib_args[i + 2] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(continuation->arg_fn_type()->ops().skip_front(FIB_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(continuation->arg(i + FIB_NUM_ARGS)), unsigned(i));
    } else {
        closure = lookup(continuation->arg(FIB_NUM_ARGS));
    }

    // allocate closure object and write values into it
    auto ptr = emit_alloca(closure_type, "fibers_closure");
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure, int lower, int upper)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_fibers";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
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
    auto kernel_fib_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), fib_type);
    irbuilder_.CreateCall(kernel_fib_func, target_args);
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    return continuation->arg(FIB_ARG_RETURN)->as_continuation();
}

enum {
    SPAWN_ARG_MEM,
    SPAWN_ARG_BODY,
    SPAWN_ARG_RETURN,
    SPAWN_NUM_ARGS
};

Continuation* CodeGen::emit_spawn(Continuation* continuation) {
    assert(continuation->num_args() >= SPAWN_NUM_ARGS && "required arguments are missing");
    auto kernel = continuation->arg(SPAWN_ARG_BODY)->as<Global>()->init()->as_continuation();
    const size_t num_kernel_args = continuation->num_args() - SPAWN_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = continuation->arg(i + SPAWN_NUM_ARGS)->type();
        par_args[i] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(continuation->arg_fn_type()->ops().skip_front(SPAWN_NUM_ARGS)));
    llvm::Value* closure = nullptr;
    if (closure_type->isStructTy()) {
        closure = llvm::UndefValue::get(closure_type);
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(continuation->arg(i + SPAWN_NUM_ARGS)), unsigned(i));
    } else {
        closure = lookup(continuation->arg(0 + SPAWN_NUM_ARGS));
    }

    // allocate closure object and write values into it
    auto ptr = irbuilder_.CreateAlloca(closure_type, nullptr);
    irbuilder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure)
    llvm::Type* wrapper_arg_types[] = { irbuilder_.getInt8PtrTy(0) };
    auto wrapper_ft = llvm::FunctionType::get(irbuilder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_spawn_thread";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
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
    auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
    irbuilder_.CreateCall(kernel_par_func, target_args);
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    // bind parameter of continuation to received handle
    auto cont = continuation->arg(SPAWN_ARG_RETURN)->as_continuation();
    emit_result_phi(cont->param(1), call);
    return cont;
}

enum {
    SYNC_ARG_MEM,
    SYNC_ARG_ID,
    SYNC_ARG_RETURN,
    SYNC_NUM_ARGS
};

Continuation* CodeGen::emit_sync(Continuation* continuation) {
    assert(continuation->num_args() == SYNC_NUM_ARGS && "wrong number of arguments");
    auto id = lookup(continuation->arg(SYNC_ARG_ID));
    runtime_->sync_thread(id);
    return continuation->arg(SYNC_ARG_RETURN)->as_continuation();
}

}

