#ifdef WFV2_SUPPORT
#include "thorin/be/llvm/llvm.h"

#include <llvm/PassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar.h>

#include <wfvInterface.h>

#include "thorin/primop.h"
#include "thorin/world.h"


namespace thorin {

enum {
    VEC_ARG_MEM,
    VEC_ARG_LENGTH,
    VEC_ARG_LOWER,
    VEC_ARG_UPPER,
    VEC_ARG_BODY,
    VEC_ARG_RETURN,
    VEC_NUM_ARGS
};

Lambda* CodeGen::emit_vectorize_continuation(Lambda* lambda) {
    auto target = lambda->to()->as_lambda();
    assert(target->intrinsic() == Intrinsic::Vectorize);
    assert(lambda->num_args() >= VEC_NUM_ARGS && "required arguments are missing");

    // arguments
    auto vector_length = lookup(lambda->arg(VEC_ARG_LENGTH));
    auto lower = lookup(lambda->arg(VEC_ARG_LOWER));
    auto upper = lookup(lambda->arg(VEC_ARG_UPPER));
    auto kernel = lambda->arg(VEC_ARG_BODY)->as<Global>()->init()->as_lambda();

    const size_t num_kernel_args = lambda->num_args() - VEC_NUM_ARGS;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_kernel_args + 1);
    simd_args[0] = builder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        Type type = lambda->arg(i + VEC_NUM_ARGS)->type();
        simd_args[i + 1] = convert(type);
    }

    auto simd_type = llvm::FunctionType::get(builder_.getVoidTy(), llvm_ref(simd_args), false);
    auto kernel_simd_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name() + "_vectorize", simd_type);

    // build iteration loop and wire the calls
    llvm::CallInst* simd_kernel_call;
    auto llvm_entry = emit_function_decl(entry_);
    create_loop(lower, upper, vector_length, llvm_entry, [&](llvm::Value* counter) {
        Array<llvm::Value*> args(num_kernel_args + 1);
        args[0] = counter; // loop index
        for (size_t i = 0; i < num_kernel_args; ++i) {
            // check target type
            Def arg = lambda->arg(i + VEC_NUM_ARGS);
            auto llvm_arg = lookup(arg);
            if (arg->type().isa<PtrType>())
                llvm_arg = builder_.CreateBitCast(llvm_arg, simd_args[i + 1]);
            args[i + 1] = llvm_arg;
        }
        // call new function
        simd_kernel_call = builder_.CreateCall(kernel_simd_func, llvm_ref(args));
    });

    u32 vector_length_constant = lambda->arg(VEC_ARG_LENGTH)->as<PrimLit>()->qu32_value();
    wfv_todo_.emplace_back(vector_length_constant, emit_function_decl(kernel), simd_kernel_call);

    return lambda->arg(VEC_ARG_RETURN)->as_lambda();
}

void CodeGen::emit_vectorize(u32 vector_length, llvm::Function* kernel_func, llvm::CallInst* simd_kernel_call) {
    // ensure proper loop forms
    FunctionPassManager pm(module_);
    pm.add(llvm::createLICMPass());
    pm.add(llvm::createLCSSAPass());
    pm.run(*kernel_func);

    // vectorize function
    auto kernel_simd_func = simd_kernel_call->getCalledFunction();
    kernel_simd_func->deleteBody();
    WFVInterface::WFVInterface wfv(module_, &context_, kernel_func, kernel_simd_func, vector_length);
    wfv.addCommonMappings(true, true, true, true, false);
    auto loop_counter_argument = kernel_func->getArgumentList().begin();
    bool b_simd = wfv.addSIMDSemantics(*loop_counter_argument, false, true, false, true, false, true);
    assert(b_simd && "simd semantics for vectorization failed");
    bool b = wfv.run();
    assert(b && "vectorization failed");

    // inline kernel
    llvm::InlineFunctionInfo info;
    llvm::InlineFunction(simd_kernel_call, info);
}

}
#endif
