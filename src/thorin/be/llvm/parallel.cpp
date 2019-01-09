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

Lam* CodeGen::emit_parallel(Lam* lam) {
    // arguments
    assert(lam->app()->num_args() >= PAR_NUM_ARGS && "required arguments are missing");
    auto num_threads = lookup(lam->app()->arg(PAR_ARG_NUMTHREADS));
    auto lower = lookup(lam->app()->arg(PAR_ARG_LOWER));
    auto upper = lookup(lam->app()->arg(PAR_ARG_UPPER));
    auto kernel = lam->app()->arg(PAR_ARG_BODY)->as<Global>()->init()->as_lam();

    const size_t num_kernel_args = lam->app()->num_args() - PAR_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args + 1);
    par_args[0] = irbuilder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = lam->app()->arg(i + PAR_NUM_ARGS)->type();
        par_args[i + 1] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lam->app()->arg()->type()->as<TupleType>()->ops().skip_front(PAR_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    if (num_kernel_args != 1) {
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = irbuilder_.CreateInsertValue(closure, lookup(lam->app()->arg(i + PAR_NUM_ARGS)), unsigned(i));
    } else {
        closure = lookup(lam->app()->arg(PAR_NUM_ARGS));
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
    auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
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

    return lam->app()->arg(PAR_ARG_RETURN)->as_lam();
}

enum {
    SPAWN_ARG_MEM,
    SPAWN_ARG_BODY,
    SPAWN_ARG_RETURN,
    SPAWN_NUM_ARGS
};

Lam* CodeGen::emit_spawn(Lam* lam) {
    assert(lam->app()->num_args() >= SPAWN_NUM_ARGS && "required arguments are missing");
    auto kernel = lam->app()->arg(SPAWN_ARG_BODY)->as<Global>()->init()->as_lam();
    const size_t num_kernel_args = lam->app()->num_args() - SPAWN_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = lam->app()->arg(i + SPAWN_NUM_ARGS)->type();
        par_args[i] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lam->app()->arg()->type()->as<TupleType>()->ops().skip_front(SPAWN_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    for (size_t i = 0; i < num_kernel_args; ++i)
        closure = irbuilder_.CreateInsertValue(closure, lookup(lam->app()->arg(i + SPAWN_NUM_ARGS)), unsigned(i));

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
    auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
    irbuilder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = irbuilder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = irbuilder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i)
        target_args[i] = irbuilder_.CreateExtractValue(val, { unsigned(i) });

    // call kernel body
    auto par_type = llvm::FunctionType::get(irbuilder_.getVoidTy(), llvm_ref(par_args), false);
    auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
    irbuilder_.CreateCall(kernel_par_func, target_args);
    irbuilder_.CreateRetVoid();

    // restore old insert point
    irbuilder_.SetInsertPoint(old_bb);

    // bind parameter of lam to received handle
    auto l = lam->app()->arg(SPAWN_ARG_RETURN)->as_lam();
    emit_result_phi(l->param(1), call);
    return l;
}

enum {
    SYNC_ARG_MEM,
    SYNC_ARG_ID,
    SYNC_ARG_RETURN,
    SYNC_NUM_ARGS
};

Lam* CodeGen::emit_sync(Lam* lam) {
    assert(lam->app()->num_args() == SYNC_NUM_ARGS && "wrong number of arguments");
    auto id = lookup(lam->app()->arg(SYNC_ARG_ID));
    runtime_->sync_thread(id);
    return lam->app()->arg(SYNC_ARG_RETURN)->as_lam();
}

}

