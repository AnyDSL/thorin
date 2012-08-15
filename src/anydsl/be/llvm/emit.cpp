#include "anydsl/be/llvm/emit.h"

#include <boost/unordered_map.hpp>

#include <llvm/Constant.h>
#include <llvm/Constants.h>
#include <llvm/Function.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Analysis/Verifier.h>
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

typedef boost::unordered_map<const Lambda*, llvm::Function*> FctMap;
typedef boost::unordered_map<const Lambda*, llvm::BasicBlock*> BBMap;

class CodeGen {
public:

    CodeGen(const World& world);

    void emit();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Function* emitFct(const Lambda* lambda);
    void emitBB(const Lambda* lambda);
    llvm::BasicBlock* lambda2bb(const Lambda* lambda);

public:

    const World& world;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
    FctMap top;
    BBMap bbs;
    const Lambda* curLam;
    llvm::Function* curFct;
    size_t retPos;
    const Param* retParam;
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
                llvm::Function* f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, lambda->debug, module);
                top.insert(std::make_pair(lambda, f));
            }

    for_all (lf, top) {
        curLam = lf.first;
        size_t retPos = curLam->pi()->nextPi();

        for_all (p, curLam->params()) {
            if (p->index() == retPos) {
                retParam = p;
                break;
            }
            anydsl_assert(p->index() <= retPos, "return param dead");
        }

        curFct = lf.second;
        emitBB(curLam);

#ifndef NDEBUG
        //llvm::verifyFunction(*f);
#endif
    }
}

void CodeGen::emitBB(const Lambda* lambda) {
    llvm::BasicBlock* bb = lambda2bb(lambda);

    if (!bb->empty())
        return;

    builder.SetInsertPoint(bb);
    std::vector<llvm::Value*> values;

#if 0
    // place phis
    for_all (param, lambda->params()) {
        llvm::PHINode* phi = builder.CreatePHI(convert(param->type()), param->phiOps().size());

        for_all (op, param->phiOps())
            phi->addIncoming(emit(op.def()), lambda2bb(op.from()));
    }
#endif

    for_all (arg, lambda->args())
        values.push_back(emit(arg));

    LambdaSet targets = lambda->targets();
    const Def* to = lambda->to();

    switch (targets.size()) {
        case 0: {
            assert(to->as<Param>());
            assert(values.size() == 1);
            builder.CreateRet(values[0]);
            return;
        }
        case 1: {
            if (const Lambda* toLambda = to->isa<Lambda>()) {
                llvm::BasicBlock* bb = lambda2bb(toLambda);
                builder.CreateBr(bb);
                emitBB(toLambda);
                return;
            }
        }
        case 2: {
            const Select* select = to->as<Select>();
            const Lambda* tLambda = select->tval()->as<Lambda>();
            const Lambda* fLambda = select->fval()->as<Lambda>();

            llvm::Value* cond = emit(select->cond());
            llvm::BasicBlock* tbb = lambda2bb(tLambda);
            llvm::BasicBlock* fbb = lambda2bb(fLambda);
            builder.CreateCondBr(cond, tbb, fbb);

            emitBB(tLambda);
            emitBB(fLambda);

            return;
        }
        default:
            ANYDSL_UNREACHABLE;
    }
}

llvm::BasicBlock* CodeGen::lambda2bb(const Lambda* lambda) {
    BBMap::iterator i = bbs.find(lambda);
    if (i != bbs.end())
        return i->second;

    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, lambda->debug, curFct);
    bbs[lambda] = bb;

    return bb;
}

void emit(const World& world) {
    CodeGen cg(world);
    cg.emit();
    cg.module->dump();
#ifndef NDEBUG
    //llvm::verifyModule(*cg.module);
#endif
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
            llvm::Type* retType = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->numelems() - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    switch (pi->numelems()) {
                        case 0:
                            retType = llvm::Type::getVoidTy(context);
                            break;
                        case 1:
                            retType = convert(pi->elem(0));
                            break;
                        default: {
                            Array<llvm::Type*> elems(pi->numelems());
                            size_t i = 0;
                            for_all (elem, pi->elems())
                                elems[i++] = convert(elem);

                            llvm::ArrayRef<llvm::Type*> structTypes(elems.begin(), elems.end());
                            retType = llvm::StructType::get(context, structTypes);
                            break;
                        }
                    }
                } else
                    elems[i++] = convert(elem);
            }

            assert(retType);
            llvm::ArrayRef<llvm::Type*> paramTypes(elems.begin(), elems.end());
            return llvm::FunctionType::get(retType, paramTypes, false);
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
        llvm::Value* tval = emit(select->tval());
        llvm::Value* fval = emit(select->fval());
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

    def->dump();
    ANYDSL_UNREACHABLE;
}

} // namespace anydsl
} // namespace be_llvm
