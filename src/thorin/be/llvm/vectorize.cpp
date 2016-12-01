#ifdef RV_SUPPORT
#include "thorin/be/llvm/llvm.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar.h>

#include <rv/rv.h>
#include <rv/transforms/loopExitCanonicalizer.h>
#include <rv/analysis/maskAnalysis.h>

#include "thorin/primop.h"
#include "thorin/util/log.h"
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

Continuation* CodeGen::emit_vectorize_continuation(Continuation* continuation) {
    auto target = continuation->callee()->as_continuation();
    assert_unused(target->intrinsic() == Intrinsic::Vectorize);
    assert(continuation->num_args() >= VEC_NUM_ARGS && "required arguments are missing");

    // arguments
    auto vector_length = lookup(continuation->arg(VEC_ARG_LENGTH));
    auto lower = lookup(continuation->arg(VEC_ARG_LOWER));
    auto upper = lookup(continuation->arg(VEC_ARG_UPPER));
    auto kernel = continuation->arg(VEC_ARG_BODY)->as<Global>()->init()->as_continuation();

    const size_t num_kernel_args = continuation->num_args() - VEC_NUM_ARGS;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_kernel_args + 1);
    simd_args[0] = irbuilder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = continuation->arg(i + VEC_NUM_ARGS)->type();
        simd_args[i + 1] = convert(type);
    }

    auto simd_type = llvm::FunctionType::get(irbuilder_.getVoidTy(), llvm_ref(simd_args), false);
    auto kernel_simd_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name() + "_vectorize", simd_type);

    // build iteration loop and wire the calls
    llvm::CallInst* simd_kernel_call;
    auto llvm_entry = emit_function_decl(entry_);
    create_loop(lower, upper, vector_length, llvm_entry, [&](llvm::Value* counter) {
        Array<llvm::Value*> args(num_kernel_args + 1);
        args[0] = counter; // loop index
        for (size_t i = 0; i < num_kernel_args; ++i) {
            // check target type
            auto arg = continuation->arg(i + VEC_NUM_ARGS);
            auto llvm_arg = lookup(arg);
            if (arg->type()->isa<PtrType>())
                llvm_arg = irbuilder_.CreateBitCast(llvm_arg, simd_args[i + 1]);
            args[i + 1] = llvm_arg;
        }
        // call new function
        simd_kernel_call = irbuilder_.CreateCall(kernel_simd_func, llvm_ref(args));
    });

    if (!continuation->arg(VEC_ARG_LENGTH)->isa<PrimLit>())
        ELOG("vector length must be hard-coded at %", continuation->arg(VEC_ARG_LENGTH)->location());
    u32 vector_length_constant = continuation->arg(VEC_ARG_LENGTH)->as<PrimLit>()->qu32_value();
    vec_todo_.emplace_back(vector_length_constant, emit_function_decl(kernel), simd_kernel_call);

    return continuation->arg(VEC_ARG_RETURN)->as_continuation();
}

void CodeGen::emit_vectorize(u32 vector_length, llvm::Function* kernel_func, llvm::CallInst* simd_kernel_call) {
    // ensure proper loop forms
    legacy::FunctionPassManager pm(module_.get());
    pm.add(llvm::createLICMPass());
    pm.add(llvm::createLCSSAPass());
    pm.run(*kernel_func);

    // vectorize function
    auto simd_kernel_func = simd_kernel_call->getCalledFunction();
    simd_kernel_func->deleteBody();

    auto rv_info = new rv::RVInfo(module_.get(), &context_, kernel_func, simd_kernel_func, vector_length, -1, false, false, false, false, nullptr);
    auto loop_counter_arg = kernel_func->getArgumentList().begin();

    rv::VectorShape res = rv::VectorShape::uni();
    rv::VectorShapeVec args;
    args.push_back(rv::VectorShape::strided(1, vector_length));
    for (auto it = std::next(loop_counter_arg), end = kernel_func->getArgumentList().end(); it != end; ++it) {
        args.push_back(rv::VectorShape::uni());
    }

    rv::VectorMapping target_mapping(kernel_func, simd_kernel_func, vector_length, -1, res, args);
    rv::VectorizationInfo vec_info(target_mapping);

    rv::VectorizerInterface vectorizer(*rv_info, kernel_func);

    // TODO: use parameters from command line
    const bool useSSE   = false;
    const bool useSSE41 = false;
    const bool useSSE42 = false;
    const bool useNEON  = false;
    const bool useAVX   = true;
    rv_info->addCommonMappings(useSSE, useSSE41, useSSE42, useAVX, useNEON);

    llvm::DominatorTree dom_tree(*kernel_func);
    llvm::PostDominatorTree pdom_tree;
    pdom_tree.runOnFunction(*kernel_func);
    llvm::LoopInfo loop_info(dom_tree);

    llvm::DFG dfg(dom_tree);
    dfg.create(*kernel_func);

    llvm::CDG cdg(*pdom_tree.DT);
    cdg.create(*kernel_func);

    LoopExitCanonicalizer canonicalizer(loop_info);
    canonicalizer.canonicalize(*kernel_func);

    vectorizer.analyze(vec_info, cdg, dfg, loop_info, pdom_tree, dom_tree);

    MaskAnalysis* mask_analysis = vectorizer.analyzeMasks(vec_info, loop_info);
    assert(mask_analysis);

    bool mask_ok = vectorizer.generateMasks(vec_info, *mask_analysis, loop_info);
    assert_unused(mask_ok);

    bool linearize_ok = vectorizer.linearizeCFG(vec_info, *mask_analysis, loop_info, pdom_tree, dom_tree);
    assert_unused(linearize_ok);

    const llvm::DominatorTree new_dom_tree(*vec_info.getMapping().scalarFn); // Control conversion does not preserve the dominance tree
    bool vectorize_ok = vectorizer.vectorize(vec_info, new_dom_tree);
    assert_unused(vectorize_ok);

    vectorizer.finalize();

    delete mask_analysis;

    // inline kernel
    llvm::InlineFunctionInfo info;
    llvm::InlineFunction(simd_kernel_call, info);
}

}
#endif
