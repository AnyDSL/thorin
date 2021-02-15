#include "opencl.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c/c.h"

namespace thorin::c_be {

OpenCLCodeGen::OpenCLCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, debug)
    , kernel_config_(kernel_config)
{}

void OpenCLCodeGen::emit(std::ostream& stream) {
    emit_c(world(), kernel_config_, stream, Lang::OPENCL, debug());
}

}
