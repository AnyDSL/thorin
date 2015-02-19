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
    auto wrapper_name = kernel->unique_name() + "_parallel";
    auto wrapper = (llvm::Function*)module_->getOrInsertFunction(wrapper_name, wrapper_ft);
    runtime_->parallel_for(num_threads, lower, upper, ptr, wrapper);

    // set insert point to the wrapper function
    auto oldBB = builder_.GetInsertBlock();
    auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
    builder_.SetInsertPoint(bb);

    // extract all arguments from the closure
    auto wrapper_args = wrapper->arg_begin();
    auto load_ptr =  builder_.CreateBitCast(&*wrapper_args, llvm::PointerType::get(closure_type, 0));
    auto val = builder_.CreateLoad(load_ptr);
    for (size_t i = 0; i < num_kernel_args; ++i)
        builder_.CreateExtractValue(val, { unsigned(i) });

    // create loop iterating over range:
    // for (int i=lower; i<upper; ++i)
    //   body(i, <closure_elems>);
    auto wrapper_lower = &*(++wrapper_args);
    auto wrapper_upper = &*(++wrapper_args);
    create_loop(wrapper_lower, wrapper_upper, builder_.getInt32(1), wrapper, [&](llvm::Value* counter) {
        std::vector<llvm::Value*> args(num_kernel_args + 1);
        args[0] = counter; // loop index
        for (size_t i = 0; i < num_kernel_args; ++i) {
            // check target type
            Def arg = lambda->arg(i + PAR_NUM_ARGS);
            auto llvm_arg = lookup(arg);
            if (arg->type().isa<PtrType>())
                llvm_arg = builder_.CreateBitCast(llvm_arg, par_args[i]);
            args[i + 1] = llvm_arg;
        }

        // call kernel body
        auto par_type = llvm::FunctionType::get(builder_.getVoidTy(), llvm_ref(par_args), false);
        auto kernel_par_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name(), par_type);
        builder_.CreateCall(kernel_par_func, args);
    });
    builder_.CreateRetVoid();

    // restore old insert point
    builder_.SetInsertPoint(oldBB);

    return lambda->arg(PAR_ARG_RETURN)->as_lambda();
}

}

