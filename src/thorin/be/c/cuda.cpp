#include "cuda.h"

namespace thorin::c {

CUDACodeGen::CUDACodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, kernel_config, Lang::CUDA, debug)
{}

}
