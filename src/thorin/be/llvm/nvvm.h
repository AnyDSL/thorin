#ifndef THORIN_BE_NVVM_H
#define THORIN_BE_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class NVVMCodeGen : public CodeGen {
public:
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
};

}

#endif
