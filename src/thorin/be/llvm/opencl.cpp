#include "thorin/be/llvm/opencl.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"


namespace thorin {

OpenCLCodeGen::OpenCLCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
    , kernel_config_(kernel_config)
{}

void OpenCLCodeGen::emit(std::ostream& stream) {
    thorin::emit_c(world(), kernel_config_, stream, Lang::OPENCL, debug());
}

}
