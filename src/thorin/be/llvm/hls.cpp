#include "thorin/be/llvm/hls.h"

#include <fstream>
#include <stdexcept>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/be/c.h"

namespace thorin {

Continuation* CodeGen::emit_hls(Continuation* continuation) {
    std::vector<llvm::Value*> args(continuation->num_args()-3);
    Continuation* ret = nullptr;
    for (size_t i = 2, j = 0; i < continuation->num_args(); ++i) {
        if (auto cont = continuation->arg(i)->isa_continuation()) {
            ret = cont;
            continue;
        }
        args[j++] = emit(continuation->arg(i));
    }
    auto callee = continuation->arg(1)->as<Global>()->init()->as_continuation();
    callee->make_exported();
    irbuilder_.CreateCall(emit_function_decl(callee), args);
    assert(ret);
    return ret;
}

HLSCodeGen::HLSCodeGen(World& world, const Cont2Config& kernel_config, int opt, bool debug)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::C, llvm::CallingConv::C, opt, debug)
    , kernel_config_(kernel_config)
{}

void HLSCodeGen::emit(std::ostream& stream) {
    thorin::emit_c(world(), kernel_config_, stream, Lang::HLS, debug());
}

}
