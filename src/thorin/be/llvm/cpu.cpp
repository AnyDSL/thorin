#include "thorin/be/llvm/cpu.h"

#include <iostream>

#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>

namespace thorin {

CPUCodeGen::CPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C)
{
    llvm::InitializeAllTargets();
    auto triple_str = llvm::sys::getDefaultTargetTriple();
    module_->setTargetTriple(triple_str);
    llvm::Triple triple(triple_str);
    llvm::EngineBuilder builder(module_);
    builder.setMCPU(llvm::sys::getHostCPUName()).setMArch(triple.getArchName());
    auto machine = builder.selectTarget();
    assert(machine);
    module_->setDataLayout(machine->getDataLayout()->getStringRepresentation());
}

}
