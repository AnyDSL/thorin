#include "thorin/be/llvm/cpu.h"

#include <llvm/Support/Host.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>


namespace thorin {

CPUCodeGen::CPUCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C)
{
    //llvm::ExecutionEngine engine(module_);
}

}
