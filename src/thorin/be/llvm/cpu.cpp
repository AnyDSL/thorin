#include "thorin/be/llvm/cpu.h"

#include <llvm/Support/Host.h>

namespace thorin {

CPUCodeGen::CPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C)
{
    module_->setTargetTriple(llvm::sys::getDefaultTargetTriple());
}

}
