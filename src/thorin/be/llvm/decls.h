#ifndef THORIN_BE_LLVM_DECLS_H
#define THORIN_BE_LLVM_DECLS_H

#include "thorin/util/autoptr.h"

namespace llvm {
    class LLVMContext;
    class Function;
    class Module;
    class Type;
}

namespace thorin {

class LLVMDecls {
public:
    LLVMDecls(llvm::LLVMContext&, llvm::Module*);

#define NVVM_DECL(fun_name) \
    llvm::Function* get_ ## fun_name() { \
        return register_in_module(fun_name ## _); \
    }
#include "nvvm_decls.h"

    llvm::Type* get_nvvm_device_ptr_type() { return nvvm_device_ptr_ty_; }

#define SPIR_DECL(fun_name) \
    llvm::Function* get_ ## fun_name() { \
        return register_in_module(fun_name ## _); \
    }
#include "spir_decls.h"

    llvm::Type* get_spir_device_ptr_type() { return spir_device_ptr_ty_; }

private:
    llvm::Function* register_in_module(llvm::Function*);

#define NVVM_DECL(fun_name) \
    llvm::Function* fun_name ## _;
#include "nvvm_decls.h"
    llvm::Type* nvvm_device_ptr_ty_;

#define SPIR_DECL(fun_name) \
    llvm::Function* fun_name ## _;
#include "spir_decls.h"
    llvm::Type* spir_device_ptr_ty_;

    llvm::Module* mod;
    AutoPtr<llvm::Module> nvvm_mod;
    AutoPtr<llvm::Module> spir_mod;
};

}
#endif
