#include "anydsl2/be/llvm.h"

#include <algorithm>
#include <boost/unordered_map.hpp>

#include <llvm/Constant.h>
#include <llvm/Constants.h>
#include <llvm/Function.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/IRBuilder.h>

#include "anydsl2/def.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/loopforest.h"
#include "anydsl2/analyses/placement.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {
namespace be_llvm {

template<class T> llvm::ArrayRef<T> llvm_ref(const Array<T>& array) {
    return llvm::ArrayRef<T>(array.begin(), array.end());
}

template<class T> llvm::ArrayRef<T> llvm_ref(const ArrayRef<T>& array) {
    return llvm::ArrayRef<T>(array.begin(), array.end());
}

//------------------------------------------------------------------------------

typedef boost::unordered_map<Lambda*, llvm::Function*> FctMap;
typedef boost::unordered_map<const Param*, llvm::Value*> ParamMap;
typedef boost::unordered_map<const Param*, llvm::PHINode*> PhiMap;
typedef boost::unordered_map<const PrimOp*, llvm::Value*> PrimOpMap;
typedef Array<llvm::BasicBlock*> BBMap;

//------------------------------------------------------------------------------

class CodeGen {
public:

    CodeGen(const World& world, EmitHook& hook);

    void emit();

    llvm::Type* map(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Value* lookup(const Def* def);

private:

    const World& world_;
    EmitHook& hook_;
    llvm::LLVMContext context_;
    llvm::IRBuilder<> builder_;
    llvm::Module* module_;
    ParamMap params_;
    PhiMap phis_;
    PrimOpMap primops_;
};

CodeGen::CodeGen(const World& world, EmitHook& hook)
    : world_(world)
    , hook_(hook)
    , context_()
    , builder_(context_)
    , module_(new llvm::Module("anydsl", context_))
{}

//------------------------------------------------------------------------------

void CodeGen::emit() {
    LambdaSet roots = find_root_lambdas(world_.lambdas());

    FctMap fcts;

    // map all root-level lambdas to llvm function stubs
    for_all (lambda, roots) {
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->debug, module_);
        fcts.insert(std::make_pair(lambda, f));
    }

    // for all top-level functions
    for_all (lf, fcts) {
        Lambda* lambda = lf.first;
        assert(lambda->is_returning());
        llvm::Function* fct = lf.second;
        assert(lambda->ho_params().size() == 1 && "unsupported number of higher order params");

        // map params
        llvm::Function::arg_iterator arg = fct->arg_begin();
        for_all (param, lambda->fo_params()) {
            arg->setName(param->debug);
            params_[param] = arg++;
        }

        const Param* ret_param = lambda->ho_params().front();
        Scope scope(lambda);
        BBMap bbs(scope.size());

        // map all bb-like lambdas to llvm bb stubs 
        for_all (lambda, scope.rpo())
            bbs[lambda->sid()] = llvm::BasicBlock::Create(context_, lambda->debug, fct);

        Array< std::vector<const PrimOp*> > places = place(scope);

        // emit body for each bb
        for_all (lambda, scope.rpo()) {
            assert(lambda == scope.entry() || lambda->is_bb());
            builder_.SetInsertPoint(bbs[lambda->sid()]);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && lambda != scope.entry()) {
                for_all (param, lambda->params())
                    phis_[param] = builder_.CreatePHI(map(param->type()), param->peek().size(), param->debug);
            }

            std::vector<const PrimOp*> primops = places[lambda->sid()];

            for_all (primop, primops)
                if (!primop->type()->isa<Pi>()) // don't touch higher-order primops
                    primops_[primop] = emit(primop);

            // terminate bb
            size_t num_targets = lambda->targets().size();
            if (num_targets == 0) {         // case 0: return
                // this is a return
                assert(lambda->to()->as<Param>() == ret_param);
                assert(lambda->args().size() == 1);
                builder_.CreateRet(lookup(lambda->arg(0)));
            } else if (num_targets == 1) {  // case 1: three sub-cases
                Lambda* tolambda = lambda->to()->as_lambda();

                if (tolambda->is_bb())      // case a) ordinary jump
                    builder_.CreateBr(bbs[tolambda->sid()]);
                else {
                    // put all first-order args into an array
                    Array<llvm::Value*> args(lambda->args().size() - 1);
                    for_all2 (&arg, args, fo_arg, lambda->fo_args())
                        arg = lookup(fo_arg);
                    llvm::CallInst* call = builder_.CreateCall(fcts[tolambda], llvm_ref(args));
                    
                    const Def* ho_arg = lambda->ho_args().front();
                    if (ho_arg == ret_param)        // case b) call + return
                        builder_.CreateRet(call); 
                    else {                          // case c) call + continuation
                        Lambda* succ = ho_arg->as_lambda();
                        params_[succ->param(0)] = call;
                        builder_.CreateBr(bbs[succ->sid()]);
                    }
                }
            } else {                        // case 2: branch
                assert(num_targets == 2);
                const Select* select = lambda->to()->as<Select>();
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bbs[select->tval()->as<Lambda>()->sid()];
                llvm::BasicBlock* fbb = bbs[select->fval()->as<Lambda>()->sid()];
                builder_.CreateCondBr(cond, tbb, fbb);
            }
        }

        // add missing arguments to phis
        for_all (p, phis_) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for_all (peek, param->peek())
                phi->addIncoming(lookup(peek.def()), bbs[peek.from()->sid()]);
        }

        params_.clear();
        phis_.clear();
        primops_.clear();
    }

    module_->dump();
    llvm::verifyModule(*this->module_);
    delete module_;
}

llvm::Value* CodeGen::lookup(const Def* def) {
    if (def->is_const())
        return emit(def);

    if (const PrimOp* primop = def->isa<PrimOp>())
        return primops_[primop];

    const Param* param = def->as<Param>();
    ParamMap::iterator i = params_.find(param);
    if (i != params_.end())
        return i->second;

    assert(phis_.find(param) != phis_.end());

    return phis_[param];
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());

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
        llvm::Value* from = lookup(conv->from());
        llvm::Type* to = map(conv->type());

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
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
    }

    if (const TupleOp* tupleop = def->isa<TupleOp>()) {
        llvm::Value* tuple = lookup(tupleop->tuple());
        unsigned idxs[1] = { tupleop->index()->primlit_value<unsigned>() };

        if (tupleop->node_kind() == Node_Extract)
            return builder_.CreateExtractValue(tuple, idxs);

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = lookup(insert->value());

        return builder_.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));

        for (unsigned i = 0, e = tuple->ops().size(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder_.CreateInsertValue(agg, lookup(tuple->op(i)), idxs);
        }

        return agg;
    }

    if (const PrimLit* primlit = def->isa<PrimLit>()) {
        llvm::Type* type = map(primlit->type());
        Box box = primlit->box();

        switch (primlit->primtype_kind()) {
            case PrimType_u1:  return builder_.getInt1(box.get_u1().get());
            case PrimType_u8:  return builder_.getInt8(box.get_u8());
            case PrimType_u16: return builder_.getInt16(box.get_u16());
            case PrimType_u32: return builder_.getInt32(box.get_u32());
            case PrimType_u64: return builder_.getInt64(box.get_u64());
            case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    // bottom and any
    if (const Undef* undef = def->isa<Undef>())
        return llvm::UndefValue::get(map(undef->type()));

    if (const Load* load = def->isa<Load>())
        return builder_.CreateLoad(lookup(load->ptr()));

    if (const Store* store = def->isa<Store>())
        return builder_.CreateStore(lookup(store->val()), lookup(store->ptr()));

    if (const Slot* slot = def->isa<Slot>())
        return builder_.CreateAlloca(map(slot->type()));

    if (const CCall* ccall = def->isa<CCall>()) {
        size_t num_args = ccall->num_args();

        Array<llvm::Type*> arg_types(num_args);
        for_all2 (&arg_type, arg_types, arg, ccall->args())
            arg_type = map(arg->type());

        Array<llvm::Value*> arg_vals(num_args);
        for_all2 (&arg_val, arg_vals, arg, ccall->args())
            arg_val = lookup(arg);

        llvm::FunctionType* ft = llvm::FunctionType::get(
                ccall->returns_void() ? llvm::Type::getVoidTy(context_) : map(ccall->rettype()),
                llvm_ref(arg_types), ccall->vararg());
        llvm::Function* callee = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, ccall->callee(), module_);
        return builder_.CreateCall(callee, llvm_ref(arg_vals));
    }

    assert(!def->is_corenode());
    return hook_.emit(def);
}

llvm::Type* CodeGen::map(const Type* type) {
    switch (type->node_kind()) {
        case Node_PrimType_u1:  return llvm::IntegerType::get(context_,  1);
        case Node_PrimType_u8:  return llvm::IntegerType::get(context_,  8);
        case Node_PrimType_u16: return llvm::IntegerType::get(context_, 16);
        case Node_PrimType_u32: return llvm::IntegerType::get(context_, 32);
        case Node_PrimType_u64: return llvm::IntegerType::get(context_, 64);
        case Node_PrimType_f32: return llvm::Type::getFloatTy(context_);
        case Node_PrimType_f64: return llvm::Type::getDoubleTy(context_);

        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* ret = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);

            for_all (elem, pi->elems()) {
                if (const Pi* pi = elem->isa<Pi>()) {
                    if (pi->empty())
                        ret = llvm::Type::getVoidTy(context_);
                    else if (pi->size() == 1)
                        ret = map(pi->elem(0));
                    else {
                        Array<llvm::Type*> elems(pi->size());
                        for_all2 (&elem, elems, pi_elem, pi->elems())
                            elem = map(pi_elem);
                        ret = llvm::StructType::get(context_, llvm_ref(elems));
                    }
                } else
                    elems[i++] = map(elem);
            }

            assert(ret);
            return llvm::FunctionType::get(ret, llvm_ref(elems), false);
        }

        case Node_Sigma: {
            // TODO watch out for cycles!
            const Sigma* sigma = type->as<Sigma>();
            Array<llvm::Type*> elems(sigma->elems().size());
            for_all2 (&elem, elems, sigma_elem, sigma->elems())
                elem = map(sigma_elem);

            return llvm::StructType::get(context_, llvm_ref(elems));
        }

        default: 
            assert(!type->is_corenode());
            return hook_.map(type);
    }
}

//------------------------------------------------------------------------------

void emit(const World& world, EmitHook& hook) {
    CodeGen cg(world, hook);
    cg.emit();
}

//------------------------------------------------------------------------------

} // namespace anydsl
} // namespace be_llvm
