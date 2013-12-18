#include "thorin/be/llvm/cpu.h"

#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>


namespace thorin {

CPUCodeGen::CPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C)
{
    module_->setTargetTriple(llvm::sys::getDefaultTargetTriple());
    //llvm::EngineBuilder builder(module_);
    //auto machine = builder.selectTarget();
    //assert(machine);
    //module_->setDataLayout(machine->getDataLayout()->getStringRepresentation());
}

}
