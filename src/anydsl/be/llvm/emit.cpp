#include "anydsl/be/llvm/emit.h"

#include <boost/scoped_array.hpp>

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

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const AIRNode* n);

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

llvm::Type* CodeGen::convert(const Type* type) {
    switch (type->indexKind()) {
        case Index_PrimType_u1:  return llvm::IntegerType::get(context, 1);
        case Index_PrimType_u8:  return llvm::IntegerType::get(context, 8);
        case Index_PrimType_u16: return llvm::IntegerType::get(context, 16);
        case Index_PrimType_u32: return llvm::IntegerType::get(context, 32);
        case Index_PrimType_u64: return llvm::IntegerType::get(context, 64);
        case Index_PrimType_f32: return llvm::IntegerType::get(context, 32);
        case Index_PrimType_f64: return llvm::IntegerType::get(context, 64);

        case Index_Pi: {
            const Pi* pi = type->as<Pi>();

            llvm::Type* retType = 0;
            size_t i = 0;

            boost::scoped_array<llvm::Type*> elems(new llvm::Type*[pi->numElems() - 1]);

            // extract "return" type, collect all other types
            for_all (t, pi->elems()) {
                if (t->isa<Pi>()) {
                    anydsl_assert(retType == 0, "not yet supported");
                    retType = convert(t);
                } else
                    elems[i++] = convert(t);
            }

            return llvm::FunctionType::get(retType, elems);
        }

        case Index_Sigma: {
            // TODO watch out for cycles!

            const Sigma* sigma = type->as<Sigma>();

            boost::scoped_array<llvm::Type*> elems(new llvm::Type*[sigma->numElems()]);
            size_t i = 0;
            for_all (t, sigma->elems())
                elems[i++] = convert(t);

            return llvm::StructType::get(context, elems);
        }

        default: ANYDSL_UNREACHABLE;
    }
}

llvm::Value* CodeGen::emit(const AIRNode* n) {
    if (!n->isCoreNode())
        ANYDSL_NOT_IMPLEMENTED;

    return 0;
#if 0
    if (n->indexKind()
    switch (n->indexKind()) {
        case Index_cmp_eq:
            builder.CreateFCmp
    }
#endif
}

}
}
