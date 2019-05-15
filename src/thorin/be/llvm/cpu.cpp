#include "thorin/be/llvm/cpu.h"

#include <cstdlib>

#include <vector>

#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

namespace thorin {

std::vector<std::string>
CPUCodeGen::GetTargets() {
  return std::vector<std::string>({
    "cortex-a53",
    "aurora"
  });
}

struct LLVMTargetDesc {
  std::string triple;
  std::string cpu;
  std::string features;

  LLVMTargetDesc(std::string triple="", std::string cpu="", std::string features="") : triple(triple), cpu(cpu), features(features) {}

  static LLVMTargetDesc
  Create(std::string llvm_cpu_target) {
    if (llvm_cpu_target == "cortex-a53") {
      return LLVMTargetDesc("aarch64-unknown-linux-gnu", "cortex-a53", "+fp-armv8 +neon +crc +crypto +sha2 +aes");
    } else if (llvm_cpu_target == "aurora") {
      return LLVMTargetDesc("ve", "ve", "");
    }

    throw std::invalid_argument("Unknown LLVM target: " + llvm_cpu_target);
  }
};


CPUCodeGen::CPUCodeGen(World& world, std::string cpu_target_name)
: CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C)
{
    llvm::InitializeNativeTarget();
    auto triple_str   = llvm::sys::getDefaultTargetTriple();
    auto cpu_str      = llvm::sys::getHostCPUName();
    std::string features_str;
    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);
    for (auto& feature : features)
        features_str += (feature.getValue() ? "+" : "-") + feature.getKey().str() + ",";

    const char* target_triple   = std::getenv("ANYDSL_TARGET_TRIPLE");
    const char* target_cpu      = std::getenv("ANYDSL_TARGET_CPU");
    const char* target_features = std::getenv("ANYDSL_TARGET_FEATURES");

    // prefer cmd line targets over env vars
    if (cpu_target_name.length() > 0) {
      auto target_desc = LLVMTargetDesc::Create(cpu_target_name);
      target_triple = target_desc.triple.c_str();
      target_cpu = target_desc.cpu.c_str();
      target_features = target_desc.features.c_str();
    }

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
    module_->setDataLayout(machine_->createDataLayout());
    module_->setTargetTriple(triple_str);
}

}
