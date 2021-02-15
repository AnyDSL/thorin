#include "hls.h"

namespace thorin::c {

HLSCodeGen::HLSCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, kernel_config, Lang::HLS, debug)
{}

}
