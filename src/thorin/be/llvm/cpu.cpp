#include "thorin/be/llvm/cpu.h"

#include <llvm/MC/TargetRegistry.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

namespace thorin::llvm {

CPUCodeGen::CPUCodeGen(Thorin& thorin, int opt, bool debug, std::string& target_triple, std::string& target_cpu, std::string& target_attr)
    : CodeGen(thorin, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
{
    llvm::InitializeNativeTarget();
    auto triple_str = llvm::sys::getDefaultTargetTriple();
    auto cpu_str    = llvm::sys::getHostCPUName();
    std::string features_str;
    auto features = llvm::sys::getHostCPUFeatures();
    for (auto& feature : features) {
        features_str += (features_str == "" ? "" : ",");
        features_str += (feature.getValue() ? "+" : "-") + feature.getKey().str();
    }

    if (!target_triple.empty() && !target_cpu.empty()) {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        triple_str   = target_triple;
        cpu_str      = target_cpu;
        features_str = target_attr;
    }

    std::string error;
    llvm::TargetOptions options;

#if LLVM_VERSION_MAJOR >= 21
    llvm::Triple triple(triple_str);

    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    assert(target && "can't create target for target architecture");

    machine_.reset(target->createTargetMachine(triple, cpu_str, features_str, options, std::nullopt));
    module().setTargetTriple(triple);
#else
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error);
    assert(target && "can't create target for target architecture");

    machine_.reset(target->createTargetMachine(triple_str, cpu_str, features_str, options, std::nullopt));
    module().setTargetTriple(triple_str);
#endif

    module().setDataLayout(machine_->createDataLayout());
}

}
