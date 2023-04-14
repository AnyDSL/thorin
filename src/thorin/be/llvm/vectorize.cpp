#include "thorin/config.h"

#if THORIN_ENABLE_RV
#include "thorin/be/llvm/llvm.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

#include <rv/rv.h>
#include <rv/vectorizationInfo.h>
#include <rv/resolver/resolvers.h>
#include <rv/passes/loopExitCanonicalizer.h>
#include <rv/passes.h>
#include <rv/region/FunctionRegion.h>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin::llvm {

struct VectorizeArgs {
    enum {
        Mem = 0,
        Length,
        Body,
        Return,
        Num
    };
};

Continuation* CodeGen::emit_vectorize_continuation(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    auto target = body->callee()->as_nom<Continuation>();
    assert_unused(target->intrinsic() == Intrinsic::Vectorize);
    assert(body->num_args() >= VectorizeArgs::Num && "required arguments are missing");

    // Important: Must emit the memory object otherwise the
    // memory operations before the call to vectorize are all gone!
    emit_unsafe(body->arg(0));

    // arguments
    auto kernel = body->arg(VectorizeArgs::Body)->as<Global>()->init()->as_nom<Continuation>();
    const size_t num_kernel_args = body->num_args() - VectorizeArgs::Num;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_kernel_args + 1);
    simd_args[0] = irbuilder.getInt32Ty(); // loop index
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto type = body->arg(i + VectorizeArgs::Num)->type();
        simd_args[i + 1] = convert(type);
    }

    auto simd_type = llvm::FunctionType::get(irbuilder.getVoidTy(), llvm_ref(simd_args), false);
    auto kernel_simd_func = (llvm::Function*)module_->getOrInsertFunction(kernel->unique_name() + "_vectorize", simd_type).getCallee()->stripPointerCasts();
    kernel_simd_func->addFnAttr("target-cpu", machine().getTargetCPU());
    kernel_simd_func->addFnAttr("target-features", machine().getTargetFeatureString());

    // build iteration loop and wire the calls
    Array<llvm::Value*> args(num_kernel_args + 1);
    args[0] = irbuilder.getInt32(0);
    for (size_t i = 0; i < num_kernel_args; ++i) {
        // check target type
        auto arg = body->arg(i + VectorizeArgs::Num);
        auto llvm_arg = emit(arg);
        if (arg->type()->isa<PtrType>())
            llvm_arg = irbuilder.CreateBitCast(llvm_arg, simd_args[i + 1]);
        args[i + 1] = llvm_arg;
    }
    auto simd_kernel_call = irbuilder.CreateCall(kernel_simd_func, llvm_ref(args));

    if (!body->arg(VectorizeArgs::Length)->isa<PrimLit>())
        world().edef(body->arg(VectorizeArgs::Length), "vector length must be known at compile-time");
    u32 vector_length_constant = body->arg(VectorizeArgs::Length)->as<PrimLit>()->qu32_value();
    vec_todo_.emplace_back(vector_length_constant, emit_fun_decl(kernel), simd_kernel_call);

    return body->arg(VectorizeArgs::Return)->as_nom<Continuation>();
}

void CodeGen::emit_vectorize(u32 vector_length, llvm::Function* kernel_func, llvm::CallInst* simd_kernel_call) {
    verify();

    // setup LLVM analysis infrastructure
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // ensure proper loop forms
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::SimplifyCFGPass());
    FPM.addPass(llvm::SROAPass());
    FPM.addPass(llvm::EarlyCSEPass());
    FPM.addPass(llvm::SCCPPass());
    FPM.addPass(llvm::FixIrreduciblePass()); // make all loops reducible (has to run first!)
    FPM.addPass(llvm::PromotePass()); // CNSPass relies on mem2reg for now

    FPM.addPass(llvm::RequireAnalysisPass<llvm::OptimizationRemarkEmitterAnalysis, llvm::Function>());

    llvm::LoopPassManager LPM;
    LPM.addPass(llvm::LICMPass());
    FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM), /*UseMemorySSA=*/true));

    FPM.addPass(llvm::LCSSAPass());

    FPM.run(*kernel_func, FAM);

    // vectorize function
    auto simd_kernel_func = simd_kernel_call->getCalledFunction();
    simd_kernel_func->deleteBody();

    auto loop_counter_arg = kernel_func->arg_begin();

    auto alignment = 1; // Be conservative and assume alignment of one byte
    rv::VectorShape res = rv::VectorShape::uni(alignment);
    rv::VectorShapeVec args;
    args.push_back(rv::VectorShape::cont(vector_length));
    for (auto it = std::next(loop_counter_arg), end = kernel_func->arg_end(); it != end; ++it) {
        args.push_back(rv::VectorShape::uni(alignment));
    }

    rv::VectorMapping target_mapping(
        kernel_func,
        simd_kernel_func,
        vector_length,
        -1,
        res,
        args,
        rv::CallPredicateMode::SafeWithoutPredicate);

    rv::FunctionRegion funcRegion(*kernel_func);
    rv::Region funcRegionWrapper(funcRegion);
    rv::VectorizationInfo vec_info(funcRegionWrapper, target_mapping);

    llvm::TargetIRAnalysis ir_analysis;
    llvm::TargetTransformInfo tti = ir_analysis.run(*kernel_func, FAM);
    llvm::TargetLibraryAnalysis lib_analysis;
    llvm::TargetLibraryInfo tli = lib_analysis.run(*kernel_func, FAM);
    rv::PlatformInfo platform_info(*module_.get(), &tti, &tli);

    if (vector_length == 1) {
        llvm::ValueToValueMapTy argMap;
        auto itCalleeArgs = simd_kernel_func->args().begin();
        auto itSourceArgs = kernel_func->args().begin();
        auto endSourceArgs = kernel_func->args().end();

        for (; itSourceArgs != endSourceArgs; ++itCalleeArgs, ++itSourceArgs) {
            argMap[&*itSourceArgs] = &*itCalleeArgs;
        }

        llvm::SmallVector<llvm::ReturnInst*,4> retVec;
        llvm::CloneFunctionInto(simd_kernel_func, kernel_func, argMap, llvm::CloneFunctionChangeType::DifferentModule, retVec);

        // lower mask intrinsics for scalar code (vector_length == 1)
        rv::lowerIntrinsics(*simd_kernel_func);
    } else {
        rv::Config config = rv::Config::createForFunction(*kernel_func);
        config.enableIRPolish = config.useAVX2;

        config.maxULPErrorBound = 35; // allow vector math with imprecision up to 3.5 ULP
        rv::addSleefResolver(config, platform_info);

        rv::VectorizerInterface vectorizer(platform_info, config);
        {
            llvm::DominatorTree dom_tree(*kernel_func);
            llvm::LoopInfo loop_info(dom_tree);
            LoopExitCanonicalizer canonicalizer(loop_info);
            canonicalizer.canonicalize(*kernel_func);
        }

        // setup LLVM analysis infrastructure
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        llvm::PassBuilder PB;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        FAM.getResult<llvm::LoopAnalysis>(*kernel_func);

        vectorizer.analyze(vec_info, FAM);

        bool lin_ok = vectorizer.linearize(vec_info, FAM);
        assert_unused(lin_ok);

        bool vectorize_ok = vectorizer.vectorize(vec_info, FAM, nullptr);
        assert_unused(vectorize_ok);

        vectorizer.finalize();
    }

    // inline kernel
    llvm::InlineFunctionInfo info;
    llvm::InlineFunction(*simd_kernel_call, info);

    // remove vectorized function
    if (simd_kernel_func->hasNUses(0))
        simd_kernel_func->eraseFromParent();
    else
        simd_kernel_func->addFnAttr(llvm::Attribute::AlwaysInline);
}

}
#endif
