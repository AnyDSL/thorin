#include "thorin/be/llvm/cpu.h"

#include <llvm/ADT/Triple.h>
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
    module_->setTargetTriple(triple_str);
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error);
    assert(target && "can't create target for host architecture");
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple_str, llvm::sys::getHostCPUName(), "" /* features */, options));
    module_->setDataLayout(machine->createDataLayout());
}

}
