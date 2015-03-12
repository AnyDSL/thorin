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

Lambda* CodeGen::emit_parallel(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Parallel);
    assert(lambda->num_args() >= PAR_NUM_ARGS && "required arguments are missing");

    // arguments
    auto num_threads = lookup(lambda->arg(PAR_ARG_NUMTHREADS));
    auto lower = lookup(lambda->arg(PAR_ARG_LOWER));
    auto upper = lookup(lambda->arg(PAR_ARG_UPPER));
    auto kernel = lambda->arg(PAR_ARG_BODY)->as<Global>()->init()->as_lambda();

    const size_t num_kernel_args = lambda->num_args() - PAR_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args + 1);
    par_args[0] = builder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        Type type = lambda->arg(i + PAR_NUM_ARGS)->type();
        par_args[i + 1] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lambda->arg_fn_type()->args().slice_from_begin(PAR_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    for (size_t i = 0; i < num_kernel_args; ++i)
        closure = builder_.CreateInsertValue(closure, lookup(lambda->arg(i + PAR_NUM_ARGS)), unsigned(i));

    // allocate closure object and write values into it
    auto ptr = builder_.CreateAlloca(closure_type, nullptr);
    builder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure, int lower, int upper)
    llvm::Type* wrapper_arg_types[] = { builder_.getInt8PtrTy(0), builder_.getInt32Ty(), builder_.getInt32Ty() };
    auto wrapper_ft = llvm::FunctionType::get(builder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_parallel_for";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
    runtime_->parallel_for(num_threads, lower, upper, ptr, wrapper);

    // set insert point to the wrapper function
    auto old_bb = builder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
    builder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = builder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = builder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args + 1);
    for (size_t i = 0; i < num_kernel_args; ++i)
        target_args[i + 1] = builder_.CreateExtractValue(val, { unsigned(i) });

    // create loop iterating over range:
    // for (int i=lower; i<upper; ++i)
    //   body(i, <closure_elems>);
    auto wrapper_lower = &*(++wrapper_args);
    auto wrapper_upper = &*(++wrapper_args);
    create_loop(wrapper_lower, wrapper_upper, builder_.getInt32(1), wrapper, [&](llvm::Value* counter) {
        // call kernel body
        target_args[0] = counter; // loop index
        auto par_type = llvm::FunctionType::get(builder_.getVoidTy(), llvm_ref(par_args), false);
        auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
        builder_.CreateCall(kernel_par_func, target_args);
    });
    builder_.CreateRetVoid();

    // restore old insert point
    builder_.SetInsertPoint(old_bb);

    return lambda->arg(PAR_ARG_RETURN)->as_lambda();
}

enum {
    SPAWN_ARG_MEM,
    SPAWN_ARG_BODY,
    SPAWN_ARG_RETURN,
    SPAWN_NUM_ARGS
};

Lambda* CodeGen::emit_spawn(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Spawn);
    assert(lambda->num_args() >= SPAWN_NUM_ARGS && "required arguments are missing");

    auto kernel = lambda->arg(SPAWN_ARG_BODY)->as<Global>()->init()->as_lambda();
    const size_t num_kernel_args = lambda->num_args() - SPAWN_NUM_ARGS;

    // build parallel-function signature
    Array<llvm::Type*> par_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i) {
        Type type = lambda->arg(i + SPAWN_NUM_ARGS)->type();
        par_args[i] = convert(type);
    }

    // fetch values and create a unified struct which contains all values (closure)
    auto closure_type = convert(world_.tuple_type(lambda->arg_fn_type()->args().slice_from_begin(SPAWN_NUM_ARGS)));
    llvm::Value* closure = llvm::UndefValue::get(closure_type);
    for (size_t i = 0; i < num_kernel_args; ++i)
        closure = builder_.CreateInsertValue(closure, lookup(lambda->arg(i + SPAWN_NUM_ARGS)), unsigned(i));

    // allocate closure object and write values into it
    auto ptr = builder_.CreateAlloca(closure_type, nullptr);
    builder_.CreateStore(closure, ptr, false);

    // create wrapper function and call the runtime
    // wrapper(void* closure)
    llvm::Type* wrapper_arg_types[] = { builder_.getInt8PtrTy(0) };
    auto wrapper_ft = llvm::FunctionType::get(builder_.getVoidTy(), wrapper_arg_types, false);
    auto wrapper_name = kernel->unique_name() + "_parallel_spawn";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
    auto tid = runtime_->parallel_spawn(ptr, wrapper);

    // set insert point to the wrapper function
    auto old_bb = builder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
    builder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr = builder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = builder_.CreateLoad(load_ptr);
    std::vector<llvm::Value*> target_args(num_kernel_args);
    for (size_t i = 0; i < num_kernel_args; ++i)
        target_args[i] = builder_.CreateExtractValue(val, { unsigned(i) });

    // call kernel body
    auto par_type = llvm::FunctionType::get(builder_.getVoidTy(), llvm_ref(par_args), false);
    auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
    builder_.CreateCall(kernel_par_func, target_args);
    builder_.CreateRetVoid();

    // restore old insert point
    builder_.SetInsertPoint(old_bb);

    // bind parameter of continuation to received handle
    auto ret = lambda->arg(SPAWN_ARG_RETURN)->as_lambda();
    params_[ret->params().back()] = tid;

    return ret;
}

enum {
    SYNC_ARG_MEM,
    SYNC_ARG_ID,
    SYNC_ARG_RETURN,
    SYNC_NUM_ARGS
};

Lambda* CodeGen::emit_sync(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Sync);
    assert(lambda->num_args() == SYNC_NUM_ARGS && "wrong number of arguments");

    auto id = lookup(lambda->arg(SYNC_ARG_ID));
    runtime_->parallel_sync(id);

    return lambda->arg(SYNC_ARG_RETURN)->as_lambda();
}

}

