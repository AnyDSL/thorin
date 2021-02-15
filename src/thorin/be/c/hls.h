#ifndef THORIN_BE_LLVM_HLS_H
#define THORIN_BE_LLVM_HLS_H

#include "c.h"

namespace thorin::c_be {

class HLSCodeGen : public CodeGen {
public:
    HLSCodeGen(World& world, const Cont2Config&, int opt, bool debug);
};

}

#endif
