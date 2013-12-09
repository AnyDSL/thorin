#include <iostream>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include "impala/type.h"

const impala::Type* llvm2impala(impala::TypeTable&, llvm::Type*);

int main() {
    auto& context = llvm::getGlobalContext();
    int num = llvm::Intrinsic::num_intrinsics - 1;
    for (int i = 1; i != num; ++i) {
        auto id = (llvm::Intrinsic::ID) i;
        
        if (!llvm::Intrinsic::isOverloaded(id)) {
            auto type = llvm::Intrinsic::getType(context, id);
            std::cout << llvm::Intrinsic::getName(id) << " ";
            std::cout.flush();
            type->dump();
            std::cout << std::endl;
        }
    }
}

const impala::Type* llvm2impala(impala::TypeTable& tt, llvm::Type* type) {
    if (auto int_type = llvm::dyn_cast<llvm::IntegerType>(type)) {
        switch (int_type->getBitWidth()) {
            case  8: return tt.type_int8();
            case 16: return tt.type_int16();
            case 32: return tt.type_int32();
            case 64: return tt.type_int64();
            default: assert(false);
        }
    }

    if (type->isFloatTy())
        return tt.type_float();
    if (type->isDoubleTy())
        return tt.type_double();

    if (auto fn = llvm::dyn_cast<llvm::FunctionType>(type)) {
        const impala::Type* ret = llvm2impala(tt, fn->getReturnType());
        std::vector<const impala::Type*> params;
        bool valid = true;
        for (size_t i = 0, e = fn->getNumParams(); i != e; ++i) {
            params[i] = llvm2impala(tt, fn->getParamType(i));
            valid &= params[i] != nullptr;
        }
    }

    return nullptr;
}
