#include "thorin/be/llvm/runtime.h"

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/SourceMgr.h>

#include "thorin/be/llvm/llvm.h"

namespace thorin {

Runtime::Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder,
                 llvm::Type* device_ptr_ty, const char* mod_name)
    : target_(target)
    , builder_(builder)
    , device_ptr_ty_(device_ptr_ty)
{
    llvm::SMDiagnostic diag;
    module_ = llvm::ParseIRFile(mod_name, diag, context);
}

llvm::Function* Runtime::get(const char* name) {
    return llvm::cast<llvm::Function>(target_->getOrInsertFunction(name, module_->getFunction(name)->getFunctionType()));
}

}
