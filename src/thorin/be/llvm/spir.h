#ifndef THORIN_BE_LLVM_SPIR_H
#define THORIN_BE_LLVM_SPIR_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class SPIRCodeGen : public CodeGen {
public:
    SPIRCodeGen(World& world);

protected:
    virtual void emit_function_decl_hook(Lambda*, llvm::Function*) override;
    virtual llvm::FunctionType* convert_fn_type(Lambda*) override;
    virtual llvm::Value* emit_mmap(const Map*) override;
    virtual std::string get_alloc_name() const override { THORIN_UNREACHABLE; /*alloc not supported in SPIR*/ }
    virtual std::string get_output_name(const std::string& name) const override { return name + ".spir"; }
    virtual std::string get_binary_output_name(const std::string& name) const override { return name + ".spir.bc"; }
};

}

#endif
