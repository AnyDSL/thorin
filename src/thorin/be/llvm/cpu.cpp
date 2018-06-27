#include "thorin/be/llvm/cpu.h"

#include <cstdlib>

#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

namespace thorin {

CPUCodeGen::CPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C)
{
    llvm::InitializeNativeTarget();
    auto triple_str = llvm::sys::getDefaultTargetTriple();
    auto cpu_str    = llvm::sys::getHostCPUName();

    char* target_triple = std::getenv("ANYDSL_TARGET_TRIPLE");
    char* target_cpu    = std::getenv("ANYDSL_TARGET_CPU");

    if (target_triple && target_cpu) {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        triple_str = target_triple;
        cpu_str    = target_cpu;
    }

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error);
    assert(target && "can't create target for target architecture");
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple_str, cpu_str, "" /* features */, options, llvm::None));
    module_->setDataLayout(machine->createDataLayout());
    module_->setTargetTriple(triple_str);
}

}
