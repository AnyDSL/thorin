#include "anydsl/be/llvm/emit.h"

#include <llvm/Constant.h>
#include <llvm/Constants.h>
#include <llvm/Function.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Support/IRBuilder.h>

#include "anydsl/def.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

namespace anydsl {
namespace be_llvm {

class CodeGen {
public:

    CodeGen(const World& world);

    void emit();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const Def* def);
    void emit(const Lambda* lambda);

private:

    const World& world;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
    LambdaSet top;
};

CodeGen::CodeGen(const World& world)
    : world(world)
    , builder(context)
    , module(new llvm::Module("anydsl", context))
{}

void CodeGen::emit() {
    for_all (def, world.defs()) {
        if (const Lambda* lambda = def->isa<Lambda>()) {
            const Pi* pi = lambda->pi();
            size_t i = 0;
            for_all (elem, pi->elems()) {
                if (elem->isa<Pi>()) {
                    top.insert(lambda);
                    anydsl_assert(i == pi->numElems() - 1, "TODO");
                    break;
                }
                ++i;
            }
        }
    }

    for_all (lambda, top) {
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(convert(lambda->type()));
        llvm::Function* f = llvm::cast<llvm::Function>(module->getOrInsertFunction(lambda->debug, ft));
        builder.SetInsertPoint(&f->getEntryBlock());
        emit(lambda);
    }
}

void CodeGen::emit(const Lambda* lambda) {
    std::vector<llvm::Value*> values;

    for_all (arg, lambda->args())
        values.push_back(emit(arg));

    LambdaSet to = lambda->to();

#if 0
    if (to.size() == 2) {
        const Select* select = lambda->todef()->as<Select>();
        llvm::Value* cond = emit(select->cond());
        //builder.CreateCondBr();
    }
#endif

}

void emit(const World& world) {
    CodeGen cg(world);
    cg.emit();
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
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* retType = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->numElems() - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    anydsl_assert(retType == 0, "not yet supported");
                    if (pi->numElems() == 0)
                        retType = llvm::Type::getVoidTy(context);
                    else {
                        anydsl_assert(pi->numElems() == 1, "TODO");
                        retType = convert(pi->get(0));
                    }
                } else
                    elems[i++] = convert(elem);
            }

            assert(retType);
            return llvm::FunctionType::get(retType, llvm::ArrayRef<llvm::Type*>(elems.begin(), elems.end()), false);
        }

        case Index_Sigma: {
            // TODO watch out for cycles!

            const Sigma* sigma = type->as<Sigma>();

            Array<llvm::Type*> elems(sigma->numElems());
            size_t i = 0;
            for_all (t, sigma->elems())
                elems[i++] = convert(t);

            return llvm::StructType::get(context, llvm::ArrayRef<llvm::Type*>(elems.begin(), elems.end()));
        }

        default: ANYDSL_UNREACHABLE;
    }
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (!def->isCoreNode())
        ANYDSL_NOT_IMPLEMENTED;

    if (const Param* param = def->isa<Param>()) {
        if (top.find(param->lambda()) == top.end()) {
            // phi function
        } else {
            llvm::Function::arg_iterator args = builder.GetInsertBlock()->getParent()->arg_begin();
            std::advance(args, param->index());

            return args;
        }
    }

    if (const PrimLit* lit = def->isa<PrimLit>()) {
        llvm::Type* type = convert(lit->type());
        Box box = lit->box();
        switch (lit->kind()) {
            case PrimLit_u1:  return builder.getInt1(box.get_u1().get());
            case PrimLit_u8:  return builder.getInt8(box.get_u8());
            case PrimLit_u16: return builder.getInt8(box.get_u16());
            case PrimLit_u32: return builder.getInt8(box.get_u32());
            case PrimLit_u64: return builder.getInt8(box.get_u64());
            case PrimLit_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimLit_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = emit(bin->lhs());
        llvm::Value* rhs = emit(bin->rhs());

        if (const RelOp* rel = def->isa<RelOp>()) {
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

        const ArithOp* arith = def->as<ArithOp>();

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

    if (const Select* select = def->isa<Select>()) {
        llvm::Value* cond = emit(select->cond());
        llvm::Value* tval = emit(select->tdef());
        llvm::Value* fval = emit(select->fdef());
        return builder.CreateSelect(cond, tval, fval);
    }

    if (const TupleOp* tupleop = def->isa<TupleOp>()) {
        llvm::Value* tuple = emit(tupleop->tuple());
        unsigned idxs[1] = { unsigned(tupleop->index()) };

        if (tupleop->indexKind() == Index_Extract)
            return builder.CreateExtractValue(tuple, idxs);

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = emit(insert->value());

        return builder.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(convert(tuple->type()));

        for (unsigned i = 0, e = tuple->numOps(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder.CreateInsertValue(agg, emit(tuple->get(i)), idxs);
        }

        return agg;
    }

    if (const Undef* undef = def->isa<Undef>())
        return llvm::UndefValue::get(convert(undef->type()));

    if (const Error* error = def->isa<Error>())
        return llvm::UndefValue::get(convert(error->type()));

    ANYDSL_UNREACHABLE;
}

} // namespace anydsl
} // namespace be_llvm
