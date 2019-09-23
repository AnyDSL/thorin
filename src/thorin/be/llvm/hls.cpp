#include "thorin/be/llvm/hls.h"

#include <fstream>
#include <stdexcept>

#include "thorin/world.h"
#include "thorin/be/c.h"

namespace thorin {

HLSCodeGen::HLSCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C)
    , kernel_config_(kernel_config)
{}

void HLSCodeGen::emit(std::ostream& stream, int /*opt*/, bool debug) {
    thorin::emit_c(world_, kernel_config_, stream, Lang::HLS, debug);
}

}
