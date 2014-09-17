#ifndef THORIN_BE_LLVM_SPIR_H
#define THORIN_BE_LLVM_SPIR_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class SPIRCodeGen : public CodeGen {
public:
    SPIRCodeGen(World& world);

protected:
    virtual llvm::Function* emit_function_decl(Lambda*);
    virtual llvm::Value* emit_mmap(Def def);
    virtual llvm::Value* emit_munmap(Def def);

    virtual std::string get_alloc_name() const { assert(false && "alloc not supported in SPIR"); }
    virtual std::string get_output_name(const std::string& name) const { return name + ".spir"; }
    virtual std::string get_binary_output_name(const std::string& name) const { return name + ".spir.bc"; }
};

}

#endif
