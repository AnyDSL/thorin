#include "thorin/be/llvm/amdgpu.h"

#include <unordered_map> // TODO don't use std::unordered_*

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin::llvm {

AMDGPUCodeGen::AMDGPUCodeGen(Thorin& thorin, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(thorin, function_calling_convention, device_calling_convention, kernel_calling_convention, opt, debug)
    , kernel_config_(kernel_config) {}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

void AMDGPUCodeGen::emit_fun_decl_hook(Continuation* continuation, llvm::Function* f) {
    auto config = kernel_config_.find(continuation);
    if (config != kernel_config_.end()) {
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            Array<llvm::Metadata*> annotation_values_wgsize(3);
            auto int32_type = llvm::IntegerType::get(context(), 32);
            annotation_values_wgsize[0] = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int32_type, std::get<0>(block)));
            annotation_values_wgsize[1] = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int32_type, std::get<1>(block)));
            annotation_values_wgsize[2] = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int32_type, std::get<2>(block)));
            f->setMetadata(llvm::StringRef("reqd_work_group_size"), llvm::MDNode::get(context(), llvm_ref(annotation_values_wgsize)));
        }
    }
}

llvm::Value* AMDGPUCodeGen::emit_global(const Global* global) {
    if (global->is_mutable())
        world().wdef(global, "AMDGPU: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

llvm::Value* AMDGPUCodeGen::emit_mathop(llvm::IRBuilder<>& irbuilder, const MathOp* mathop) {
    auto make_key = [] (MathOpTag tag, unsigned bitwidth) { return (static_cast<unsigned>(tag) << 16) | bitwidth; };
    static const std::unordered_map<unsigned, std::string> ocml_functions = {
#define MATH_FUNCTION(name) \
        { make_key(MathOp_##name, 32), "__ocml_" #name "_f32" }, \
        { make_key(MathOp_##name, 64), "__ocml_" #name "_f64" },
        MATH_FUNCTION(fabs)
        MATH_FUNCTION(copysign)
        MATH_FUNCTION(round)
        MATH_FUNCTION(floor)
        MATH_FUNCTION(ceil)
        MATH_FUNCTION(fmin)
        MATH_FUNCTION(fmax)
        MATH_FUNCTION(cos)
        MATH_FUNCTION(sin)
        MATH_FUNCTION(tan)
        MATH_FUNCTION(acos)
        MATH_FUNCTION(asin)
        MATH_FUNCTION(atan)
        MATH_FUNCTION(atan2)
        MATH_FUNCTION(sqrt)
        MATH_FUNCTION(cbrt)
        MATH_FUNCTION(pow)
        MATH_FUNCTION(exp)
        MATH_FUNCTION(exp2)
        MATH_FUNCTION(log)
        MATH_FUNCTION(log2)
        MATH_FUNCTION(log10)
#undef MATH_FUNCTION
    };
    auto key = make_key(mathop->mathop_tag(), num_bits(mathop->type()->primtype_tag()));
    auto call = call_math_function(irbuilder, mathop, ocml_functions.at(key));
    llvm::cast<llvm::CallInst>(call)->setCallingConv(function_calling_convention_);
    return call;
}

llvm::Value* AMDGPUCodeGen::emit_reserve(llvm::IRBuilder<>& irbuilder, const Continuation* continuation) {
    return emit_reserve_shared(irbuilder, continuation, true);
}

llvm::Value* AMDGPUCodeGen::emit_local_memory(llvm::IRBuilder<>& irbuilder, const Continuation* continuation) {
    return emit_local_memory_base_ptr(irbuilder, continuation);
}

}
