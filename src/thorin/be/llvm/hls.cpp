#include "thorin/be/llvm/hls.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"

namespace thorin::llvm_be {

HLSCodeGen::HLSCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
    , kernel_config_(kernel_config)
{}

void HLSCodeGen::emit(std::ostream& stream) {
    thorin::emit_c(world(), kernel_config_, stream, Lang::HLS, debug());
}

}
