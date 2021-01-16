#include "thorin/be/llvm/cuda.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"

namespace thorin {

CUDACodeGen::CUDACodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
    , kernel_config_(kernel_config)
{}

void CUDACodeGen::emit(std::ostream& stream) {
    thorin::emit_c(world(), kernel_config_, stream, Lang::CUDA, debug());
}

}
