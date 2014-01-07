#ifdef WFV2_SUPPORT
#include "thorin/be/llvm/llvm.h"

#include <llvm/PassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Scalar.h>

#include "thorin/literal.h"
#include "thorin/world.h"

#include <wfvInterface.h>

namespace thorin {

static llvm::Function* get_vectorize_tid(llvm::Module* module) {
    const char* vectorize_intrinsic_name = "wfv_get_tid";
    return module->getFunction(vectorize_intrinsic_name);
}

Lambda* CodeGen::emit_vectorized(llvm::Function* current, Lambda* lambda) {
    Lambda* target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::Vectorize));
    assert(lambda->num_args() > 5 && "required arguments are missing");

    // vector length
    u32 count = lambda->arg(1)->as<PrimLit>()->qu32_value();
    u32 vector_length = lambda->arg(2)->as<PrimLit>()->qu32_value();
    assert(vector_length >= 4 && "vector_length >= 4");

    auto kernel = lambda->arg(3)->as<Global>()->init()->as_lambda();
    auto kernel_func = fcts_[kernel];
    auto ret = lambda->arg(4)->as_lambda();

    const size_t arg_index = 5;
    const size_t num_args = lambda->num_args() - arg_index;

    // build simd-function signature
    Array<llvm::Type*> simd_args(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        const Type* type = lambda->arg(i + arg_index)->type();
        simd_args[i] = map(type);
    }

    llvm::FunctionType* simd_type = llvm::FunctionType::get(builder_.getVoidTy(), llvm_ref(simd_args), false);
    llvm::Function* kernel_simd_func = (llvm::Function*)module_->getOrInsertFunction(kernel->name + "_vectorized", simd_type);

    // build iteration loop and wire the calls
    llvm::BasicBlock* header = llvm::BasicBlock::Create(context_, "vec_header", current);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(context_, "vec_body", current);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(context_, "vec_exit", current);
    // create loop phi and connect init value
    llvm::PHINode* loop_counter = llvm::PHINode::Create(builder_.getInt32Ty(), 2U, "vector_loop_phi", header);
    llvm::Value* i = builder_.getInt32(0);
    loop_counter->addIncoming(i, builder_.GetInsertBlock());
    // connect header
    builder_.CreateBr(header);
    builder_.SetInsertPoint(header);
    // create conditional branch
    llvm::Value* cond = builder_.CreateICmpSLT(loop_counter, builder_.getInt32(count));
    builder_.CreateCondBr(cond, body, exit);
    // set body
    builder_.SetInsertPoint(body);
    Array<llvm::Value*> args(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        // check target type
        Def arg = lambda->arg(i + arg_index);
        llvm::Value* llvm_arg = lookup(arg);
        if (arg->type()->isa<Ptr>())
            llvm_arg = builder_.CreateBitCast(llvm_arg, simd_args[i]);
        args[i] = llvm_arg;
    }
    // call new function
    llvm::CallInst* kernel_call = builder_.CreateCall(kernel_simd_func, llvm_ref(args));
    // inc loop counter
    loop_counter->addIncoming(builder_.CreateAdd(loop_counter, builder_.getInt32(vector_length)), body);
    builder_.CreateBr(header);
    // create branch to return
    builder_.SetInsertPoint(exit);

    // ensure proper loop forms
    FunctionPassManager pm(module_);
    pm.add(llvm::createLICMPass());
    pm.add(llvm::createLCSSAPass());
    pm.run(*kernel_func);

    // vectorize function
    auto vector_tid_getter = get_vectorize_tid(module_);
    WFVInterface::WFVInterface wfv(module_, &context_, kernel_func, kernel_simd_func, vector_length);
    if(vector_tid_getter) {
        bool b_simd = wfv.addSIMDSemantics(*vector_tid_getter, false, true, false, false, false, true, false, true, false, true);
        assert(b_simd && "simd semantics for vectorization failed");
    }
    bool b = wfv.run();
    assert(b && "vectorization failed");

    // inline kernel
    llvm::InlineFunctionInfo info;
    llvm::InlineFunction(kernel_call, info);

    // wire loop counter
    if(vector_tid_getter) {
        std::vector<llvm::CallInst*> calls;
        for (auto it = vector_tid_getter->use_begin(), e = vector_tid_getter->use_end(); it != e; ++it) {
            if (auto call = llvm::dyn_cast<llvm::CallInst>(*it))
                if (const Function* func = call->getParent()->getParent())
                    if (func == current)
                        calls.push_back(call);
        }
        for (auto it = calls.rbegin(), e = calls.rend(); it != e; ++it) {
            BasicBlock::iterator ii(*it);
            ReplaceInstWithValue((*it)->getParent()->getInstList(), ii, loop_counter);
        }
    }

    // remember to remove functions
    fcts_to_remove_.insert(kernel_func);
    fcts_to_remove_.insert(kernel_simd_func);
    fcts_to_remove_.insert(vector_tid_getter);

    return ret;
}

}

#endif
