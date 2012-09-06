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
#include "anydsl/analyses/find_root_lambdas.h"
#include "anydsl/analyses/domtree.h"
#include "anydsl/analyses/placement.h"
#include "anydsl/util/array.h"

namespace anydsl {
namespace be_llvm {

typedef boost::unordered_map<const Lambda*, llvm::Function*> FctMap;
typedef boost::unordered_map<const Lambda*, llvm::BasicBlock*> BBMap;
typedef boost::unordered_map<const Param*, llvm::Value*> ParamMap;

class CodeGen {
public:

    CodeGen(const World& world);

    void emit();

    llvm::Type* convert(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Function* emit_fct(const Lambda* lambda);
    void emitBB(const Lambda* lambda);
    llvm::BasicBlock* lambda2bb(const Lambda* lambda);
    //void recEmit(Dominators& dom, const Lambda* lambda);

private:

    const World& world_;
    llvm::LLVMContext context_;
    llvm::IRBuilder<> builder_;
    llvm::Module* module_;
    FctMap top_;
    BBMap bbs_;
    ParamMap params_;
    const Lambda* curLam_;
    llvm::Function* curFct_;
    const Param* retParam_;
};

CodeGen::CodeGen(const World& world)
    : world_(world)
    , context_()
    , builder_(context_)
    , module_(new llvm::Module("anydsl", context_))
{}

void CodeGen::emit() {
    LambdaSet roots = find_root_lambdas(world_.lambdas());

    for_all (lambda, roots) {
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(convert(lambda->type()));
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, lambda->debug, module_);
        top_.insert(std::make_pair(lambda, f));
    }

    for_all (lf, top_) {
        curLam_ = lf.first;
        size_t retPos = curLam_->pi()->ho_begin();

        for_all (p, curLam_->params()) {
            if (p->index() == retPos) {
                retParam_ = p;
                break;
            }
            anydsl_assert(p->index() <= retPos, "return param dead");
        }

        curFct_ = lf.second;
        params_.clear();

        //recEmit(dom, curLam_);

#ifndef NDEBUG
        //llvm::verifyFunction(*f);
#endif
    }

    module_->dump();
#ifndef NDEBUG
    //llvm::verifyModule(*cg.module);
#endif
}

#if 0
void CodeGen::recEmit(Dominators& dom, const Lambda* lambda) {
    Dominators::Range range = dom.children().equal_range(lambda);
    for (Dominators::DomChildren::const_iterator i = range.first, e = range.second; i != e; ++i)
        if (const Lambda* lchild = i->second->isa<Lambda>())
            emitBB(lchild);

    for (Dominators::DomChildren::const_iterator i = range.first, e = range.second; i != e; ++i)
        if (const Lambda* lchild = i->second->isa<Lambda>())
            recEmit(dom, lchild);
}
#endif


void CodeGen::emitBB(const Lambda* lambda) {
    llvm::BasicBlock* bb = lambda2bb(lambda);

    if (!bb->empty())
        return;

    builder_.SetInsertPoint(bb);
    std::vector<llvm::Value*> values;

    if (lambda == curLam_) {
        llvm::Function::arg_iterator arg = curFct_->arg_begin();
        for_all (param, lambda->params())
            if (!param->type()->isa<Pi>())
                params_[param] = arg++;
    } else {
        // place phis
        for_all (param, lambda->params()) {
            llvm::PHINode* phi = builder_.CreatePHI(convert(param->type()), param->phiOps().size());

            for_all (op, param->phiOps())
                phi->addIncoming(emit(op.def()), lambda2bb(op.from()));

            params_[param] = phi;
        }
    }

    for_all (arg, lambda->args())
        values.push_back(emit(arg));

    Lambdas targets = lambda->targets();
    const Def* to = lambda->to();

    switch (targets.size()) {
        case 0: {
            const Param* param = to->as<Param>();

            if (param == retParam_) {
                // ret
                assert(values.size() == 1);
                builder_.CreateRet(values[0]);
                return;
            } else {
                ANYDSL_UNREACHABLE;
            }
        }
        case 1: {
            if (const Lambda* toLambda = to->isa<Lambda>()) {
                llvm::BasicBlock* bb = lambda2bb(toLambda);
                builder_.CreateBr(bb);
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
            builder_.CreateCondBr(cond, tbb, fbb);

            return;
        }
        default:
            ANYDSL_UNREACHABLE;
    }
}

llvm::BasicBlock* CodeGen::lambda2bb(const Lambda* lambda) {
    BBMap::iterator i = bbs_.find(lambda);
    if (i != bbs_.end())
        return i->second;

    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context_, lambda->debug, curFct_);
    bbs_[lambda] = bb;

    return bb;
}

void emit(const World& world) {
    CodeGen cg(world);
    cg.emit();
}

llvm::Type* CodeGen::convert(const Type* type) {
    switch (type->node_kind()) {
        case Node_PrimType_u1:  return llvm::IntegerType::get(context_, 1);
        case Node_PrimType_u8:  return llvm::IntegerType::get(context_, 8);
        case Node_PrimType_u16: return llvm::IntegerType::get(context_, 16);
        case Node_PrimType_u32: return llvm::IntegerType::get(context_, 32);
        case Node_PrimType_u64: return llvm::IntegerType::get(context_, 64);
        case Node_PrimType_f32: return llvm::Type::getFloatTy(context_);
        case Node_PrimType_f64: return llvm::Type::getDoubleTy(context_);

        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* retType = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    switch (pi->size()) {
                        case 0:
                            retType = llvm::Type::getVoidTy(context_);
                            break;
                        case 1:
                            retType = convert(pi->elem(0));
                            break;
                        default: {
                            Array<llvm::Type*> elems(pi->size());
                            size_t i = 0;
                            for_all (elem, pi->elems())
                                elems[i++] = convert(elem);

                            llvm::ArrayRef<llvm::Type*> structTypes(elems.begin(), elems.end());
                            retType = llvm::StructType::get(context_, structTypes);
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

            return llvm::StructType::get(context_, llvm::ArrayRef<llvm::Type*>(elems.begin(), elems.end()));
        }

        default: ANYDSL_UNREACHABLE;
    }
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (!def->is_corenode())
        ANYDSL_NOT_IMPLEMENTED;

    if (const Param* param = def->isa<Param>()) {
        ParamMap::iterator i = params_.find(param);
        anydsl_assert(i != params_.end(), "not found");
        return i->second;
    }

    if (const PrimLit* lit = def->isa<PrimLit>()) {
        llvm::Type* type = convert(lit->type());
        Box box = lit->box();
        switch (lit->primtype_kind()) {
            case PrimType_u1:  return builder_.getInt1(box.get_u1().get());
            case PrimType_u8:  return builder_.getInt8(box.get_u8());
            case PrimType_u16: return builder_.getInt16(box.get_u16());
            case PrimType_u32: return builder_.getInt32(box.get_u32());
            case PrimType_u64: return builder_.getInt64(box.get_u64());
            case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = emit(bin->lhs());
        llvm::Value* rhs = emit(bin->rhs());

        if (const RelOp* rel = def->isa<RelOp>()) {
            switch (rel->relop_kind()) {
                case RelOp_cmp_eq:   return builder_.CreateICmpEQ (lhs, rhs);
                case RelOp_cmp_ne:   return builder_.CreateICmpNE (lhs, rhs);

                case RelOp_cmp_ugt:  return builder_.CreateICmpUGT(lhs, rhs);
                case RelOp_cmp_uge:  return builder_.CreateICmpUGE(lhs, rhs);
                case RelOp_cmp_ult:  return builder_.CreateICmpULT(lhs, rhs);
                case RelOp_cmp_ule:  return builder_.CreateICmpULE(lhs, rhs);

                case RelOp_cmp_sgt:  return builder_.CreateICmpSGT(lhs, rhs);
                case RelOp_cmp_sge:  return builder_.CreateICmpSGE(lhs, rhs);
                case RelOp_cmp_slt:  return builder_.CreateICmpSLT(lhs, rhs);
                case RelOp_cmp_sle:  return builder_.CreateICmpSLE(lhs, rhs);

                case RelOp_fcmp_oeq: return builder_.CreateFCmpOEQ(lhs, rhs);
                case RelOp_fcmp_one: return builder_.CreateFCmpONE(lhs, rhs);

                case RelOp_fcmp_ogt: return builder_.CreateFCmpOGT(lhs, rhs);
                case RelOp_fcmp_oge: return builder_.CreateFCmpOGE(lhs, rhs);
                case RelOp_fcmp_olt: return builder_.CreateFCmpOLT(lhs, rhs);
                case RelOp_fcmp_ole: return builder_.CreateFCmpOLE(lhs, rhs);

                case RelOp_fcmp_ueq: return builder_.CreateFCmpUEQ(lhs, rhs);
                case RelOp_fcmp_une: return builder_.CreateFCmpUNE(lhs, rhs);

                case RelOp_fcmp_ugt: return builder_.CreateFCmpUGT(lhs, rhs);
                case RelOp_fcmp_uge: return builder_.CreateFCmpUGE(lhs, rhs);
                case RelOp_fcmp_ult: return builder_.CreateFCmpULT(lhs, rhs);
                case RelOp_fcmp_ule: return builder_.CreateFCmpULE(lhs, rhs);

                case RelOp_fcmp_uno: return builder_.CreateFCmpUNO(lhs, rhs);
                case RelOp_fcmp_ord: return builder_.CreateFCmpORD(lhs, rhs);
            }
        }

        const ArithOp* arith = def->as<ArithOp>();

        switch (arith->arithop_kind()) {
            case ArithOp_add:  return builder_.CreateAdd (lhs, rhs);
            case ArithOp_sub:  return builder_.CreateSub (lhs, rhs);
            case ArithOp_mul:  return builder_.CreateMul (lhs, rhs);
            case ArithOp_udiv: return builder_.CreateUDiv(lhs, rhs);
            case ArithOp_sdiv: return builder_.CreateSDiv(lhs, rhs);
            case ArithOp_urem: return builder_.CreateURem(lhs, rhs);
            case ArithOp_srem: return builder_.CreateSRem(lhs, rhs);

            case ArithOp_fadd: return builder_.CreateFAdd(lhs, rhs);
            case ArithOp_fsub: return builder_.CreateFSub(lhs, rhs);
            case ArithOp_fmul: return builder_.CreateFMul(lhs, rhs);
            case ArithOp_fdiv: return builder_.CreateFDiv(lhs, rhs);
            case ArithOp_frem: return builder_.CreateFRem(lhs, rhs);

            case ArithOp_and:  return builder_.CreateAnd (lhs, rhs);
            case ArithOp_or:   return builder_.CreateOr  (lhs, rhs);
            case ArithOp_xor:  return builder_.CreateXor (lhs, rhs);

            case ArithOp_shl:  return builder_.CreateShl (lhs, rhs);
            case ArithOp_lshr: return builder_.CreateLShr(lhs, rhs);
            case ArithOp_ashr: return builder_.CreateAShr(lhs, rhs);
        }
    }

    if (const ConvOp* conv = def->isa<ConvOp>()) {
        llvm::Value* from = emit(conv->from());
        llvm::Type* to = convert(conv->type());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:  return builder_.CreateTrunc  (from, to);
            case ConvOp_zext:   return builder_.CreateZExt   (from, to);
            case ConvOp_sext:   return builder_.CreateSExt   (from, to);
            case ConvOp_stof:   return builder_.CreateSIToFP (from, to);
            case ConvOp_utof:   return builder_.CreateSIToFP (from, to);
            case ConvOp_ftrunc: return builder_.CreateFPTrunc(from, to);
            case ConvOp_ftos:   return builder_.CreateFPToSI (from, to);
            case ConvOp_ftou:   return builder_.CreateFPToUI (from, to);
            case ConvOp_fext:   return builder_.CreateFPExt  (from, to);
            case ConvOp_bitcast:return builder_.CreateBitCast(from, to);
        }
    }

    if (const Select* select = def->isa<Select>()) {
        llvm::Value* cond = emit(select->cond());
        llvm::Value* tval = emit(select->tval());
        llvm::Value* fval = emit(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
    }

    if (const TupleOp* tupleop = def->isa<TupleOp>()) {
        llvm::Value* tuple = emit(tupleop->tuple());
        unsigned idxs[1] = { unsigned(tupleop->index()) };

        if (tupleop->node_kind() == Node_Extract)
            return builder_.CreateExtractValue(tuple, idxs);

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = emit(insert->value());

        return builder_.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(convert(tuple->type()));

        for (unsigned i = 0, e = tuple->ops().size(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder_.CreateInsertValue(agg, emit(tuple->op(i)), idxs);
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
