#include "opencl.h"

namespace thorin::c_be {

OpenCLCodeGen::OpenCLCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, kernel_config, Lang::OPENCL, debug)
{}

}
