#include "thorin/be/llvm/decls.h"

#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;

namespace thorin {

LLVMDecls::LLVMDecls(LLVMContext& context, Module* mod)
    : mod(mod)
#define NVVM_DECL(fun_name) \
    , fun_name ## _(nullptr)
#include "nvvm_decls.h"
#define SPIR_DECL(fun_name) \
    , fun_name ## _(nullptr)
#include "spir_decls.h"
{
    llvm::SMDiagnostic errors;
    nvvm_mod = llvm::ParseIRFile("nvvm.s", errors, context);
    if (nvvm_mod != nullptr) {
#define NVVM_DECL(fun_name) \
        fun_name ## _ = nvvm_mod->getFunction(#fun_name);
#include "nvvm_decls.h"
    }
    nvvm_device_ptr_ty_ = IntegerType::getInt64Ty(context);

    spir_mod = llvm::ParseIRFile("spir.s", errors, context);
    if (spir_mod != nullptr) {
#define SPIR_DECL(fun_name) \
        fun_name ## _ = spir_mod->getFunction(#fun_name);
#include "spir_decls.h"
    }
    spir_device_ptr_ty_ = IntegerType::getInt64Ty(context);
}

llvm::Function* LLVMDecls::register_in_module(llvm::Function* fun) {
    assert(fun != nullptr && "fun undefined");
    return (llvm::Function*)mod->getOrInsertFunction(fun->getName(), fun->getFunctionType());
}

}
