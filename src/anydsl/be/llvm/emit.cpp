#include "anydsl/be/llvm/emit.h"

#include <boost/unordered_map.hpp>

#include <llvm/Constant.h>
#include <llvm/Constants.h>
#include <llvm/Function.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/LLVMContext.h>

#include "anydsl/def.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

namespace anydsl {
namespace be_llvm {

typedef boost::unordered_map<const Lambda*, llvm::Function*> FctMap;

class CodeGen {
public:

    CodeGen(const World& world);

    void emit();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Function* emitFct(const Lambda* lambda);
    llvm::BasicBlock* emitBB(const Lambda* lambda);

private:

    const World& world;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
    FctMap top;
};

CodeGen::CodeGen(const World& world)
    : world(world)
    , builder(context)
    , module(new llvm::Module("anydsl", context))
{}

void CodeGen::emit() {
    for_all (def, world.defs())
        if (const Lambda* lambda = def->isa<Lambda>())
            if (lambda->pi()->isHigherOrder()) {
                llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(convert(lambda->type()));
                llvm::Function* f = llvm::cast<llvm::Function>(module->getOrInsertFunction(lambda->debug, ft));
                top.insert(std::make_pair(lambda, f));
            }

    for_all (p, top) {
        const Lambda* lambda = p.first;
        llvm::Function* f = p.second;
        builder.SetInsertPoint(&f->getEntryBlock());
        emit(lambda);
    }
}

llvm::BasicBlock* CodeGen::emitBB(const Lambda* lambda) {
    std::vector<llvm::Value*> values;

    for_all (arg, lambda->args())
        values.push_back(emit(arg));

    LambdaSet to = lambda->to();

    if (to.size() == 2) {
        const Select* select = lambda->todef()->as<Select>();
        llvm::Value* cond = emit(select->cond());
        llvm::BasicBlock* tbb = emitBB(select->tdef()->as<Lambda>());
        llvm::BasicBlock* fbb = emitBB(select->fdef()->as<Lambda>());
        builder.CreateCondBr(cond, tbb, fbb);
    }
}

void emit(const World& world) {
    CodeGen cg(world);
    cg.emit();
}

llvm::Type* CodeGen::convert(const Type* type) {
    switch (type->node_kind()) {
        case Node_PrimType_u1:  return llvm::IntegerType::get(context, 1);
        case Node_PrimType_u8:  return llvm::IntegerType::get(context, 8);
        case Node_PrimType_u16: return llvm::IntegerType::get(context, 16);
        case Node_PrimType_u32: return llvm::IntegerType::get(context, 32);
        case Node_PrimType_u64: return llvm::IntegerType::get(context, 64);
        case Node_PrimType_f32: return llvm::Type::getFloatTy(context);
        case Node_PrimType_f64: return llvm::Type::getDoubleTy(context);

        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            size_t num = pi->elems().size();
            llvm::Type* retType = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(num - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    anydsl_assert(retType == 0, "not yet supported");
                    if (num == 0)
                        retType = llvm::Type::getVoidTy(context);
                    else {
                        anydsl_assert(num == 1, "TODO");
                        retType = convert(pi->elem(0));
                    }
                } else
                    elems[i++] = convert(elem);
            }

            assert(retType);
            return llvm::FunctionType::get(retType, llvm::ArrayRef<llvm::Type*>(elems.begin(), elems.end()), false);
        }

        case Node_Sigma: {
            // TODO watch out for cycles!

            const Sigma* sigma = type->as<Sigma>();

            Array<llvm::Type*> elems(sigma->elems().size());
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
        switch (lit->primtype_kind()) {
            case PrimType_u1:  return builder.getInt1(box.get_u1().get());
            case PrimType_u8:  return builder.getInt8(box.get_u8());
            case PrimType_u16: return builder.getInt16(box.get_u16());
            case PrimType_u32: return builder.getInt32(box.get_u32());
            case PrimType_u64: return builder.getInt64(box.get_u64());
            case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = emit(bin->lhs());
        llvm::Value* rhs = emit(bin->rhs());

        if (const RelOp* rel = def->isa<RelOp>()) {
            switch (rel->relop_kind()) {
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

        switch (arith->arithop_kind()) {
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

    if (const ConvOp* conv = def->isa<ConvOp>()) {
        llvm::Value* from = emit(conv->from());
        llvm::Type* to = convert(conv->to());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:  return builder.CreateTrunc  (from, to);
            case ConvOp_zext:   return builder.CreateZExt   (from, to);
            case ConvOp_sext:   return builder.CreateSExt   (from, to);
            case ConvOp_stof:   return builder.CreateSIToFP (from, to);
            case ConvOp_utof:   return builder.CreateSIToFP (from, to);
            case ConvOp_ftrunc: return builder.CreateFPTrunc(from, to);
            case ConvOp_ftos:   return builder.CreateFPToSI (from, to);
            case ConvOp_ftou:   return builder.CreateFPToUI (from, to);
            case ConvOp_bitcast:return builder.CreateBitCast(from, to);
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

        if (tupleop->node_kind() == Node_Extract)
            return builder.CreateExtractValue(tuple, idxs);

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = emit(insert->value());

        return builder.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(convert(tuple->type()));

        for (unsigned i = 0, e = tuple->ops().size(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder.CreateInsertValue(agg, emit(tuple->op(i)), idxs);
        }

        return agg;
    }

    // bottom and any
    if (const Undef* undef = def->isa<Undef>())
        return llvm::UndefValue::get(convert(undef->type()));

    ANYDSL_UNREACHABLE;
}

} // namespace anydsl
} // namespace be_llvm
