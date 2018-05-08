#include "thorin/be/llvm/opencl.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"


namespace thorin {

OpenCLCodeGen::OpenCLCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C)
    , kernel_config_(kernel_config)
{}

void OpenCLCodeGen::emit(std::ostream& stream, int /*opt*/, bool debug) {
    thorin::emit_c(world_, kernel_config_, stream, Lang::OPENCL, debug);
}

}
