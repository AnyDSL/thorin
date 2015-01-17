#include "thorin/be/llvm/llvm.h"

namespace thorin {

Lambda* CodeGen::emit_parallel_continuation(Lambda* lambda) {
    auto &world = lambda->world();
    // to-target is the desired parallel-target call
    // target(mem, num_threads, body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Parallel);
    assert(lambda->num_args() > 3 && "required arguments are missing");

    // get input
    auto num_threads  = lookup(lambda->arg(1));
    auto func = lambda->arg(2)->as<Global>()->init()->as_lambda();
    auto ret = lambda->arg(3)->as_lambda();
    const auto arg_index = 4;

    // call parallel runtime function
    auto target_fun = fcts_[func];
    llvm::Value* handle;
    if (lambda->num_args() > arg_index) {
        // fetch values and create a unified struct which contains all values (closure)
        auto closure_type = convert(world.tuple_type(lambda->arg_fn_type()->args().slice_from_begin(4)));
        llvm::Value* closure = llvm::UndefValue::get(closure_type);
        for (size_t i = arg_index, e = lambda->num_args(); i != e; ++i)
            closure = builder_.CreateInsertValue(closure, lookup(lambda->arg(i)), unsigned(i - arg_index));

        // allocate closure object and write values into it
        AutoPtr<llvm::DataLayout> dl(new llvm::DataLayout(module_));
        auto ptr = builder_.CreateAlloca(closure_type, nullptr);
        builder_.CreateStore(closure, ptr, false);

        // create wrapper function that extracts all arguments from the closure
        auto ft = llvm::FunctionType::get(builder_.getVoidTy(), { builder_.getInt8PtrTy(0) }, false);
        auto wrapper_name = lambda->unique_name() + "_parallel";
        auto wrapper = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, wrapper_name);

        handle = runtime_->parallel_create(num_threads, ptr, dl->getTypeAllocSize(closure_type), wrapper);

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
        handle = runtime_->parallel_create(num_threads, llvm::ConstantPointerNull::get(builder_.getInt8PtrTy()), 0, target_fun);
    }

    // bind parameter of continuation to received handle
    if (ret->num_params() == 1)
        params_[ret->param(0)] = handle;

    return ret;
}

void CodeGen::emit_parallel(u32, llvm::Function*, llvm::CallInst*) {
    // TODO
}

}

