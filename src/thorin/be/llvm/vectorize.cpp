#ifdef RV_SUPPORT
#include "thorin/be/llvm/llvm.h"

#include <llvm/IR/Dominators.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Passes/PassBuilder.h>

#include <rv/rv.h>
#include <rv/vectorizationInfo.h>
#include <rv/sleefLibrary.h>
#include <rv/transform/loopExitCanonicalizer.h>
#include <rv/analysis/maskAnalysis.h>
#include <rv/passes.h>

#include "thorin/primop.h"
#include "thorin/util/log.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

struct VectorizeArgs {
    enum {
        Mem = 0,
        Length,
        Align,
        Lower,
        Upper,
        Body,
        Return,
        Num
    };
};

Continuation* CodeGen::emit_vectorize_continuation(Continuation* continuation) {
    auto target = continuation->callee()->as_continuation();
    assert_unused(target->intrinsic() == Intrinsic::Vectorize);
    assert(continuation->num_args() >= VectorizeArgs::Num && "required arguments are missing");

    // arguments
    auto vector_length = lookup(continuation->arg(VectorizeArgs::Length));
    auto lower = lookup(continuation->arg(VectorizeArgs::Lower));
    auto upper = lookup(continuation->arg(VectorizeArgs::Upper));
    auto kernel = continuation->arg(VectorizeArgs::Body)->as<Global>()->init()->as_continuation();
    const size_t num_kernel_args = continuation->num_args() - VectorizeArgs::Num;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_kernel_args + 1);
    simd_args[0] = irbuilder_.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = continuation->arg(i + VectorizeArgs::Num)->type();
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
            auto arg = continuation->arg(i + VectorizeArgs::Num);
            auto llvm_arg = lookup(arg);
            if (arg->type()->isa<PtrType>())
                llvm_arg = irbuilder_.CreateBitCast(llvm_arg, simd_args[i + 1]);
            args[i + 1] = llvm_arg;
        }
        // call new function
        simd_kernel_call = irbuilder_.CreateCall(kernel_simd_func, llvm_ref(args));
    });

    if (!continuation->arg(VectorizeArgs::Length)->isa<PrimLit>())
        ELOG(continuation->arg(VectorizeArgs::Length), "vector length must be hard-coded");
    u32 vector_length_constant = continuation->arg(VectorizeArgs::Length)->as<PrimLit>()->qu32_value();
    u32 alignment_constant     = continuation->arg(VectorizeArgs::Align )->as<PrimLit>()->qu32_value();
    vec_todo_.emplace_back(vector_length_constant, alignment_constant, emit_function_decl(kernel), simd_kernel_call);

    return continuation->arg(VectorizeArgs::Return)->as_continuation();
}

void CodeGen::emit_vectorize(u32 vector_length, u32 alignment, llvm::Function* kernel_func, llvm::CallInst* simd_kernel_call) {
    // ensure proper loop forms
    legacy::FunctionPassManager pm(module_.get());
    pm.add(llvm::createLICMPass());
    pm.add(llvm::createLCSSAPass());
    pm.add(llvm::createLowerSwitchPass());
    pm.add(rv::createCNSPass()); // make all loops reducible
    pm.run(*kernel_func);

    // vectorize function
    auto simd_kernel_func = simd_kernel_call->getCalledFunction();
    simd_kernel_func->deleteBody();

    auto loop_counter_arg = kernel_func->getArgumentList().begin();

    rv::VectorShape res = rv::VectorShape::uni(alignment);
    rv::VectorShapeVec args;
    args.push_back(rv::VectorShape::cont(alignment));
    for (auto it = std::next(loop_counter_arg), end = kernel_func->getArgumentList().end(); it != end; ++it) {
        args.push_back(rv::VectorShape::uni(alignment));
    }

    rv::VectorMapping target_mapping(kernel_func, simd_kernel_func, vector_length, -1, res, args);
    rv::VectorizationInfo vec_info(target_mapping);

    llvm::FunctionAnalysisManager FAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    llvm::TargetIRAnalysis ir_analysis;
    llvm::TargetTransformInfo tti = ir_analysis.run(*kernel_func, FAM);
    llvm::TargetLibraryAnalysis lib_analysis;
    llvm::TargetLibraryInfo tli = lib_analysis.run(*kernel_func->getParent(), MAM);
    rv::PlatformInfo platform_info(*module_.get(), &tti, &tli);

    // TODO: use parameters from command line
    const bool useSSE = false;
    const bool useAVX = true;
    const bool useAVX2 = false;
    const bool impreciseFunctions = true;
    rv::addSleefMappings(useSSE, useAVX, useAVX2, platform_info, impreciseFunctions);

    rv::VectorizerInterface vectorizer(platform_info);

    llvm::DominatorTree dom_tree(*kernel_func);
    llvm::PostDominatorTree pdom_tree;
    pdom_tree.recalculate(*kernel_func);
    llvm::LoopInfo loop_info(dom_tree);

    llvm::DFG dfg(dom_tree);
    dfg.create(*kernel_func);

    llvm::CDG cdg(pdom_tree);
    cdg.create(*kernel_func);

    llvm::ScalarEvolutionAnalysis SEA;
    auto SE = SEA.run(*kernel_func, FAM);

    llvm::MemoryDependenceAnalysis MDA;
    auto MD = MDA.run(*kernel_func, FAM);

    LoopExitCanonicalizer canonicalizer(loop_info);
    canonicalizer.canonicalize(*kernel_func);

    vectorizer.analyze(vec_info, cdg, dfg, loop_info, pdom_tree, dom_tree);

    bool lin_ok = vectorizer.linearize(vec_info, cdg, dfg, loop_info, pdom_tree, dom_tree);
    assert_unused(lin_ok);

    llvm::DominatorTree new_dom_tree(*vec_info.getMapping().scalarFn); // Control conversion does not preserve the dominance tree
    bool vectorize_ok = vectorizer.vectorize(vec_info, new_dom_tree, loop_info, SE, MD, nullptr);
    assert_unused(vectorize_ok);

    vectorizer.finalize();

    // inline kernel
    llvm::InlineFunctionInfo info;
    llvm::InlineFunction(simd_kernel_call, info);

    // remove vectorized function
    if (simd_kernel_func->hasNUses(0))
        simd_kernel_func->eraseFromParent();
}

}
#endif
