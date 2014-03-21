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
    //llvm::InitializeAllTargets();
    auto triple_str = llvm::sys::getDefaultTargetTriple();
    module_->setTargetTriple(triple_str);
    // TODO how can I retrieve this magic string automatically?
    module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128");
    //llvm::Triple triple(triple_str);
    //llvm::EngineBuilder builder(module_);
    //builder.setMCPU(llvm::sys::getHostCPUName()).setMArch(triple.getArchName());
    //auto machine = builder.selectTarget();
    //assert(machine);
    //module_->setDataLayout(machine->getDataLayout()->getStringRepresentation());
}

}
