#include "thorin/be/llvm/cpu.h"

#include <cstdlib>

#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

namespace thorin::llvm {

CPUCodeGen::CPUCodeGen(World& world, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
{
    llvm::InitializeNativeTarget();
    auto triple_str   = llvm::sys::getDefaultTargetTriple();
    auto cpu_str      = llvm::sys::getHostCPUName();
    std::string features_str;
    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);
    for (auto& feature : features)
        features_str += (feature.getValue() ? "+" : "-") + feature.getKey().str() + ",";

    char* target_triple   = std::getenv("ANYDSL_TARGET_TRIPLE");
    char* target_cpu      = std::getenv("ANYDSL_TARGET_CPU");
    char* target_features = std::getenv("ANYDSL_TARGET_FEATURES");

    if (target_triple && target_cpu) {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        triple_str   = target_triple;
        cpu_str      = target_cpu;
        features_str = target_features ? target_features : "";
    }

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error);
    assert(target && "can't create target for target architecture");
    llvm::TargetOptions options;
    machine_.reset(target->createTargetMachine(triple_str, cpu_str, features_str, options, llvm::None));
    module().setDataLayout(machine_->createDataLayout());
    module().setTargetTriple(triple_str);
}

}
