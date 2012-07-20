#include "anydsl/be/llvm/emit.h"

#include <boost/scoped_array.hpp>

#include <llvm/Module.h>
#include <llvm/Function.h>
#include <llvm/Type.h>
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

    void findTopLevelFunctions();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const AIRNode* n);

private:

    const World& world;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
};

CodeGen::CodeGen(const World& world)
    : world(world)
    , builder(context)
    , module(new llvm::Module("anydsl", context))
{}

void CodeGen::findTopLevelFunctions() {
    LambdaSet top;

    for_all (def, world.defs()) {
        if (const Lambda* lambda = def->isa<Lambda>()) {
            for_all (param, lambda->params()) {
            }
        }
    }
}

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
        case Index_PrimType_f32: return llvm::Type::getFloatTy(context);
        case Index_PrimType_f64: return llvm::Type::getDoubleTy(context);

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

    if (const RelOp* rel = n->isa<RelOp>()) {
        llvm::Value* lhs = emit(rel->lhs());
        llvm::Value* rhs = emit(rel->rhs());

        switch (rel->relOpKind()) {
            case RelOp_cmp_eq:   return builder.CreateICmpEQ (lhs, rhs);
            case RelOp_cmp_ne:   return builder.CreateICmpNE (lhs, rhs);

            case RelOp_cmp_ugt:  return builder.CreateICmpUGT(lhs, rhs);
            case RelOp_cmp_uge:  return builder.CreateICmpUGE(lhs, rhs);
            case RelOp_cmp_ult:  return builder.CreateICmpULT(lhs, rhs);
            case RelOp_cmp_ule:  return builder.CreateICmpULE(lhs, rhs);

            case RelOp_cmp_sgt:  return builder.CreateICmpSGT(lhs, rhs);
            case RelOp_cmp_sge:  return builder.CreateICmpSGE(lhs, rhs);
            case RelOp_cmp_slt:  return builder.CreateICmpSLT(lhs, rhs);
            case RelOp_cmp_sle:  return builder.CreateICmpSLE(lhs, rhs);

            case RelOp_fcmp_oeq: return builder.CreateFCmpOEQ(lhs, rhs);
            case RelOp_fcmp_one: return builder.CreateFCmpONE(lhs, rhs);

            case RelOp_fcmp_ogt: return builder.CreateFCmpOGT(lhs, rhs);
            case RelOp_fcmp_oge: return builder.CreateFCmpOGE(lhs, rhs);
            case RelOp_fcmp_olt: return builder.CreateFCmpOLT(lhs, rhs);
            case RelOp_fcmp_ole: return builder.CreateFCmpOLE(lhs, rhs);

            case RelOp_fcmp_ueq: return builder.CreateFCmpUEQ(lhs, rhs);
            case RelOp_fcmp_une: return builder.CreateFCmpUNE(lhs, rhs);

            case RelOp_fcmp_ugt: return builder.CreateFCmpUGT(lhs, rhs);
            case RelOp_fcmp_uge: return builder.CreateFCmpUGE(lhs, rhs);
            case RelOp_fcmp_ult: return builder.CreateFCmpULT(lhs, rhs);
            case RelOp_fcmp_ule: return builder.CreateFCmpULE(lhs, rhs);

            case RelOp_fcmp_uno: return builder.CreateFCmpUNO(lhs, rhs);
            case RelOp_fcmp_ord: return builder.CreateFCmpORD(lhs, rhs);
        }
    }

    if (const ArithOp* arith = n->isa<ArithOp>()) {
        llvm::Value* lhs = emit(arith->lhs());
        llvm::Value* rhs = emit(arith->rhs());

        switch (arith->arithOpKind()) {
            case ArithOp_add:  return builder.CreateAdd (lhs, rhs);
            case ArithOp_sub:  return builder.CreateSub (lhs, rhs);
            case ArithOp_mul:  return builder.CreateMul (lhs, rhs);
            case ArithOp_udiv: return builder.CreateUDiv(lhs, rhs);
            case ArithOp_sdiv: return builder.CreateSDiv(lhs, rhs);
            case ArithOp_urem: return builder.CreateURem(lhs, rhs);
            case ArithOp_srem: return builder.CreateSRem(lhs, rhs);

            case ArithOp_fadd: return builder.CreateFAdd(lhs, rhs);
            case ArithOp_fsub: return builder.CreateFSub(lhs, rhs);
            case ArithOp_fmul: return builder.CreateFMul(lhs, rhs);
            case ArithOp_fdiv: return builder.CreateFDiv(lhs, rhs);
            case ArithOp_frem: return builder.CreateFRem(lhs, rhs);

            case ArithOp_and:  return builder.CreateAnd (lhs, rhs);
            case ArithOp_or:   return builder.CreateOr  (lhs, rhs);
            case ArithOp_xor:  return builder.CreateXor (lhs, rhs);

            case ArithOp_shl:  return builder.CreateShl (lhs, rhs);
            case ArithOp_lshr: return builder.CreateLShr(lhs, rhs);
            case ArithOp_ashr: return builder.CreateAShr(lhs, rhs);
        }
    }
}

} // namespace anydsl
} // namespace be_llvm
