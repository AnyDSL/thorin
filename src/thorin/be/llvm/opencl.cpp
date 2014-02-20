#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/be/c.h"
#include "thorin/be/llvm/opencl.h"

#include <iostream>
#include <fstream>

namespace thorin {

OpenCLCodeGen::OpenCLCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C)
{}

void OpenCLCodeGen::emit() {
    std::ofstream file(world_.name() + ".cl");
    if (!file.is_open())
        throw std::runtime_error("cannot write '" + world_.name() + ".cl': " + strerror(errno));
    thorin::emit_c(world_, file, OPENCL);
    file.close();
}

}
