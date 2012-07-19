#include "anydsl/be/llvm/emit.h"

#include <llvm/Module.h>
#include <llvm/Function.h>
#include <llvm/Support/IRBuilder.h>

#include "anydsl/def.h"
#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"

namespace anydsl {
namespace be_llvm {

class CodeGen {
public:

    CodeGen(const World& world);

    llvm::Type* convert(const anydsl::Type* type);

private:

    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
};

CodeGen::CodeGen(const World& world)
    : builder(context)
    , module(new llvm::Module("anydsl", context))
{}

void emit(const World& world) {
    CodeGen cg(world);
}

llvm::Type* CodeGen::convert(const anydsl::Type* type) {
    switch (type->indexKind()) {
        case Index_PrimType_u1:  return llvm::IntegerType::get(context, 1);
        case Index_PrimType_u8:  return llvm::IntegerType::get(context, 8);
        case Index_PrimType_u16: return llvm::IntegerType::get(context, 16);
        case Index_PrimType_u32: return llvm::IntegerType::get(context, 32);
        case Index_PrimType_u64: return llvm::IntegerType::get(context, 64);
        case Index_PrimType_f32: return llvm::IntegerType::get(context, 32);
        case Index_PrimType_f64: return llvm::IntegerType::get(context, 64);

        case Index_Pi: {
            //llvm::FunctionType* ft = llvm::FunctionType::get();
        }
        default: ANYDSL_UNREACHABLE;

    }
}

void emit(const anydsl::AIRNode* n) {
    if (!n->isStandardNode())
        ANYDSL_NOT_IMPLEMENTED;

    //switch (n->indexKind()) {
        //case 
    //}
}

}
}
