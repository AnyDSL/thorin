#include "thorin/be/llvm/cuda.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"

namespace thorin {

CUDACodeGen::CUDACodeGen(World& world, const Lam2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C)
    , kernel_config_(kernel_config)
{}

void CUDACodeGen::emit(std::ostream& stream, int /*opt*/, bool debug) {
    thorin::emit_c(world_, kernel_config_, stream, Lang::CUDA, debug);
}

}
