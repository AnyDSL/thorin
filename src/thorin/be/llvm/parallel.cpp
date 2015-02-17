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

Lambda* CodeGen::emit_parallel_continuation(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Parallel);
    assert(lambda->num_args() >= PAR_NUM_ARGS && "required arguments are missing");

    // arguments
    auto num_threads = lookup(lambda->arg(PAR_ARG_NUMTHREADS));
    auto lower = lookup(lambda->arg(PAR_ARG_LOWER));
    auto upper = lookup(lambda->arg(PAR_ARG_UPPER));
    auto kernel = lambda->arg(PAR_ARG_BODY)->as<Global>()->init()->as_lambda();
    auto ret = lambda->arg(PAR_ARG_RETURN)->as_lambda();

    const size_t num_kernel_args = lambda->num_args() - PAR_NUM_ARGS;

    // call parallel runtime function
    auto target_fun = fcts_[kernel];
    llvm::Value* handle;
    if (num_kernel_args) {
        // fetch values and create a unified struct which contains all values (closure)
        auto closure_type = convert(world_.tuple_type(lambda->arg_fn_type()->args().slice_from_begin(PAR_NUM_ARGS)));
        llvm::Value* closure = llvm::UndefValue::get(closure_type);
        for (size_t i = 0; i < num_kernel_args; ++i)
            closure = builder_.CreateInsertValue(closure, lookup(lambda->arg(i + PAR_NUM_ARGS)), unsigned(i));

        // allocate closure object and write values into it
        auto ptr = builder_.CreateAlloca(closure_type, nullptr);
        builder_.CreateStore(closure, ptr, false);

        // create wrapper function that extracts all arguments from the closure
        auto ft = llvm::FunctionType::get(builder_.getVoidTy(), { builder_.getInt8PtrTy(0) }, false);
        auto wrapper_name = lambda->unique_name() + "_parallel";
        auto wrapper = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, wrapper_name);

        auto layout = llvm::DataLayout(module_->getDataLayout());
        handle = runtime_->parallel_for(num_threads, lower, upper, ptr, layout.getTypeAllocSize(closure_type), wrapper);

        auto oldBB = builder_.GetInsertBlock();

        // emit wrapper function
        auto bb = llvm::BasicBlock::Create(context_, wrapper_name, wrapper);
        builder_.SetInsertPoint(bb);

        // load value in different thread and extract data from the closure
        auto load_ptr =  builder_.CreateBitCast(&*wrapper->arg_begin(), llvm::PointerType::get(closure_type, 0));
        auto val = builder_.CreateLoad(load_ptr);

        std::vector<llvm::Value*> target_args;
        for (size_t i = 0; i < num_kernel_args; ++i)
            target_args.push_back(builder_.CreateExtractValue(val, { unsigned(i) }));
        builder_.CreateCall(target_fun, target_args);
        builder_.CreateRetVoid();

        // restore old insert point
        builder_.SetInsertPoint(oldBB);
    } else {
        // no closure required
        handle = runtime_->parallel_for(num_threads, lower, upper, llvm::ConstantPointerNull::get(builder_.getInt8PtrTy()), 0, target_fun);
    }

    // bind parameter of continuation to received handle
    if (ret->num_params() == 1)
        params_[ret->param(0)] = handle;

    return ret;
}

void CodeGen::emit_parallel(llvm::Value*, llvm::Function*, llvm::CallInst*) {
    // TODO
}

}

