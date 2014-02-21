#ifndef THORIN_BE_LLVM_SPIR_H
#define THORIN_BE_LLVM_SPIR_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class SPIRCodeGen : public CodeGen {
public:
    SPIRCodeGen(World& world);

protected:
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Value* emit_memmap(Def def);

    virtual std::string get_output_name(const std::string& name) const { return name + ".spir"; }
    virtual std::string get_binary_output_name(const std::string& name) const { return name + ".spir.bc"; }
};

}

#endif
