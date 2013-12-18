#ifndef THORIN_BE_SPIR_H
#define THORIN_BE_SPIR_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class SPIRCodeGen : public CodeGen {
public:
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
};

}

#endif

