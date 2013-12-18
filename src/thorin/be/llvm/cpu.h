#include "thorin/be/llvm/cpu.h"

namespace thorin {

llvm::Function* emit_function_decl(std::string& name, Lambda* lambda) {
    llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(cg.map(lambda->type()));
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, module_);
}

}
