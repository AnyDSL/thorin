#include "anydsl2/be/llvm.h"

#include <algorithm>
#include <iostream>
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

    CodeGen(World& world, EmitHook& hook);

    void emit();

    llvm::Type* map(const Type* type);
    llvm::Value* emit(const Def* def);
    llvm::Value* lookup(const Def* def);

private:

    World& world;
    EmitHook& hook;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
    ParamMap params;
    PhiMap phis;
    PrimOpMap primops;
};

CodeGen::CodeGen(World& world, EmitHook& hook)
    : world(world)
    , hook(hook)
    , context()
    , builder(context)
    , module(new llvm::Module("anydsl", context))
{
    hook.assign(&builder, module);
}

//------------------------------------------------------------------------------

void CodeGen::emit() {
    FctMap fcts;

    // map all root-level lambdas to llvm function stubs
    for_all (lambda, find_root_lambdas(world)) {
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module);
        fcts.insert(std::make_pair(lambda, f));
    }

    // for all top-level functions
    for_all (lf, fcts) {
        Lambda* lambda = lf.first;
        assert(lambda->is_returning());
        llvm::Function* fct = lf.second;

        // map params
        const Param* ret_param = 0;
        llvm::Function::arg_iterator arg = fct->arg_begin();
        for_all (param, lambda->params()) {
            if (param->type()->isa<Mem>())
                continue;
            if (param->order() == 0) {
                arg->setName(param->name);
                params[param] = arg++;
            } else {
                assert(!ret_param);
                ret_param = param;
            }
        }

        Scope scope(lambda);
        BBMap bbs(scope.size());

        // map all bb-like lambdas to llvm bb stubs
        for_all (lambda, scope.rpo())
            bbs[lambda->sid()] = llvm::BasicBlock::Create(context, lambda->name, fct);

        Array< std::vector<const PrimOp*> > places = place(scope);

        // emit body for each bb
        for_all (lambda, scope.rpo()) {
            assert(lambda == scope.entry() || lambda->is_basicblock());
            builder.SetInsertPoint(bbs[lambda->sid()]);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && lambda != scope.entry()) {
                for_all (param, lambda->params())
                    if (!param->type()->isa<Mem>())
                        phis[param] = builder.CreatePHI(map(param->type()), param->peek().size(), param->name);
            }

            std::vector<const PrimOp*> schedule = places[lambda->sid()];
            for_all (primop, schedule)
                // if this primop is not a function, not a memory argument and not a frame
                // we have to skip it; however, CCall nodes have to be emitted.
                if (!primop->type()->isa<Pi>() && !primop->type()->isa<Frame>() 
                        && (!primop->type()->isa<Mem>() || primop->isa<CCall>() || primop->isa<Store>()))
                    primops[primop] = emit(primop);

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: builder.CreateRetVoid(); break;
                    case 1: builder.CreateRet(lookup(lambda->arg(0))); break;
                    case 2: {
                        if (lambda->arg(0)->type()->isa<Mem>()) {
                            builder.CreateRet(lookup(lambda->arg(1)));
                            break;
                        } else if (lambda->arg(1)->type()->isa<Mem>()) {
                            builder.CreateRet(lookup(lambda->arg(0)));
                            break;
                        }
                        // FALLTHROUGH
                    }
                    default: {
                        Array<llvm::Value*> values(num_args);
                        Array<llvm::Type*> elems(num_args);

                        size_t n = 0;
                        for (size_t a = 0; a < num_args; ++a) {
                            if (!lambda->arg(n)->type()->isa<Mem>()) {
                                llvm::Value* val = lookup(lambda->arg(a));
                                values[n] = val;
                                elems[n++] = val->getType();
                            }
                        }

                        assert(n == num_args || n+1 == num_args);
                        values.shrink(n);
                        elems.shrink(n);
                        llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context, llvm_ref(elems)));

                        for (size_t i = 0; i != n; ++i) {
                            unsigned idxs[1] = { unsigned(i) };
                            agg = builder.CreateInsertValue(agg, values[i], idxs);
                        }
                        builder.CreateRet(agg);
                        break;
                    }
                }
            } else if (const Select* select = lambda->to()->isa<Select>()) { // conditional branch
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bbs[select->tval()->as<Lambda>()->sid()];
                llvm::BasicBlock* fbb = bbs[select->fval()->as<Lambda>()->sid()];
                builder.CreateCondBr(cond, tbb, fbb);
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())      // ordinary jump
                    builder.CreateBr(bbs[to_lambda->sid()]);
                else {
                    // put all first-order args into an array
                    Array<llvm::Value*> args(lambda->args().size() - 1);
                    size_t i = 0;
                    const Def* ret_arg = 0;
                    for_all (arg, lambda->args())
                        if (arg->order() == 0) {
                            if (!arg->type()->isa<Mem>())
                                args[i++] = lookup(arg);
                        } else {
                            assert(!ret_arg);
                            ret_arg = arg;
                        }
                    args.shrink(i);
                    llvm::CallInst* call = builder.CreateCall(fcts[to_lambda], llvm_ref(args));

                    if (ret_arg == ret_param)       // call + return
                        builder.CreateRet(call);
                    else {                          // call + continuation
                        Lambda* succ = ret_arg->as_lambda();

                        if (llvm::StructType* st = llvm::dyn_cast<llvm::StructType>(call->getType())) {
                            // extract multiple return values from struct
                            for (unsigned i = 0, e = st->getNumElements(); i != e; ++i) {
                                unsigned idxs[1] = { i };
                                params[succ->param(i)] = builder.CreateExtractValue(call, idxs);
                            }
                        } else
                            params[succ->param(0)->type()->isa<Mem>() ? succ->param(1) : succ->param(0)] = call;

                        builder.CreateBr(bbs[succ->sid()]);
                    }
                }
            }
        }

        // add missing arguments to phis
        for_all (p, phis) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for_all (peek, param->peek())
                phi->addIncoming(lookup(peek.def()), bbs[peek.from()->sid()]);
        }

        params.clear();
        phis.clear();
        primops.clear();
    }

    module->dump();
    llvm::verifyModule(*this->module);
    delete module;
}

llvm::Value* CodeGen::lookup(const Def* def) {
    if (def->is_const())
        return emit(def);

    if (const PrimOp* primop = def->isa<PrimOp>())
        return primops[primop];

    const Param* param = def->as<Param>();
    ParamMap::iterator i = params.find(param);
    if (i != params.end())
        return i->second;

    assert(phis.find(param) != phis.end());

    return phis[param];
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (const BinOp* bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());

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

        switch (def->as<ArithOp>()->arithop_kind()) {
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
        llvm::Value* from = lookup(conv->from());
        llvm::Type* to = map(conv->type());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:    return builder.CreateTrunc   (from, to);
            case ConvOp_zext:     return builder.CreateZExt    (from, to);
            case ConvOp_sext:     return builder.CreateSExt    (from, to);
            case ConvOp_stof:     return builder.CreateSIToFP  (from, to);
            case ConvOp_utof:     return builder.CreateSIToFP  (from, to);
            case ConvOp_ftrunc:   return builder.CreateFPTrunc (from, to);
            case ConvOp_ftos:     return builder.CreateFPToSI  (from, to);
            case ConvOp_ftou:     return builder.CreateFPToUI  (from, to);
            case ConvOp_fext:     return builder.CreateFPExt   (from, to);
            case ConvOp_bitcast:  return builder.CreateBitCast (from, to);
            case ConvOp_inttoptr: return builder.CreateIntToPtr(from, to);
            case ConvOp_ptrtoint: return builder.CreatePtrToInt(from, to);
        }
    }

    if (const Select* select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder.CreateSelect(cond, tval, fval);
    }

    if (const TupleOp* tupleop = def->isa<TupleOp>()) {
        llvm::Value* tuple = lookup(tupleop->tuple());
        unsigned idxs[1] = { tupleop->index()->primlit_value<unsigned>() };

        if (tupleop->node_kind() == Node_Extract) {
            // check for CCall result
            if (tupleop->tuple()->isa<CCall>() || tupleop->tuple()->isa<Load>())
                return tuple;
            return builder.CreateExtractValue(tuple, idxs);
        }

        const Insert* insert = def->as<Insert>();
        llvm::Value* value = lookup(insert->value());

        return builder.CreateInsertValue(tuple, value, idxs);
    }

    if (const Tuple* tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));

        for (unsigned i = 0, e = tuple->ops().size(); i != e; ++i) {
            unsigned idxs[1] = { unsigned(i) };
            agg = builder.CreateInsertValue(agg, lookup(tuple->op(i)), idxs);
        }

        return agg;
    }

    if (const PrimLit* primlit = def->isa<PrimLit>()) {
        llvm::Type* type = map(primlit->type());
        Box box = primlit->box();

        switch (primlit->primtype_kind()) {
            case PrimType_u1:  return builder.getInt1(box.get_u1().get());
            case PrimType_u8:  return builder.getInt8(box.get_u8());
            case PrimType_u16: return builder.getInt16(box.get_u16());
            case PrimType_u32: return builder.getInt32(box.get_u32());
            case PrimType_u64: return builder.getInt64(box.get_u64());
            case PrimType_f32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_f64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    // bottom and any
    if (const Undef* undef = def->isa<Undef>())
        return llvm::UndefValue::get(map(undef->type()));

    if (const Load* load = def->isa<Load>())
        return builder.CreateLoad(lookup(load->ptr()));

    if (const Store* store = def->isa<Store>())
        return builder.CreateStore(lookup(store->val()), lookup(store->ptr()));

    if (const Slot* slot = def->isa<Slot>())
        return builder.CreateAlloca(map(slot->type()->as<Ptr>()->ref()), 0, slot->unique_name());

    if (const CCall* ccall = def->isa<CCall>()) {
        size_t num_args = ccall->num_args();

        Array<llvm::Type*> arg_types(num_args);
        for_all2 (&arg_type, arg_types, arg, ccall->args())
            arg_type = map(arg->type());

        Array<llvm::Value*> arg_vals(num_args);
        for_all2 (&arg_val, arg_vals, arg, ccall->args())
            arg_val = lookup(arg);

        llvm::FunctionType* ft = llvm::FunctionType::get(
                ccall->returns_void() ? llvm::Type::getVoidTy(context) : map(ccall->rettype()),
                llvm_ref(arg_types), ccall->vararg());
        llvm::Function* callee = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, ccall->callee(), module);
        return builder.CreateCall(callee, llvm_ref(arg_vals));
    }

    if (def->isa<Enter>() || def->isa<Leave>())
        return 0;

    assert(!def->is_corenode());
    return hook.emit(def);
}

llvm::Type* CodeGen::map(const Type* type) {
    assert(!type->isa<Mem>());
    switch (type->node_kind()) {
        case Node_PrimType_u1:  return llvm::IntegerType::get(context,  1);
        case Node_PrimType_u8:  return llvm::IntegerType::get(context,  8);
        case Node_PrimType_u16: return llvm::IntegerType::get(context, 16);
        case Node_PrimType_u32: return llvm::IntegerType::get(context, 32);
        case Node_PrimType_u64: return llvm::IntegerType::get(context, 64);
        case Node_PrimType_f32: return llvm::Type::getFloatTy(context);
        case Node_PrimType_f64: return llvm::Type::getDoubleTy(context);
        case Node_Ptr:          return llvm::PointerType::getUnqual(map(type->as<Ptr>()->ref()));

        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* ret = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);
            for_all (elem, pi->elems()) {
                if (elem->isa<Mem>())
                    continue;
                if (const Pi* pi = elem->isa<Pi>()) {
                    assert(!ret && "only one 'return' supported");
                    if (pi->empty())
                        ret = llvm::Type::getVoidTy(context);
                    else if (pi->size() == 1)
                        ret = pi->elem(0)->isa<Mem>() ? llvm::Type::getVoidTy(context) : map(pi->elem(0));
                    else if (pi->size() == 2) {
                        if (pi->elem(0)->isa<Mem>())
                            ret = map(pi->elem(1));
                        else if (pi->elem(1)->isa<Mem>())
                            ret = map(pi->elem(0));
                        else
                            goto multiple;
                    } else {
multiple:
                        Array<llvm::Type*> elems(pi->size());
                        size_t j = 0;
                        for_all2 (&elem, elems, pi_elem, pi->elems()) {
                            if(pi_elem->isa<Mem>())
                                continue;
                            ++j;
                            elem = map(pi_elem);
                        }
                        elems.shrink(j);
                        ret = llvm::StructType::get(context, llvm_ref(elems));
                    }
                } else
                    elems[i++] = map(elem);
            }
            elems.shrink(i);
            assert(ret);

            return llvm::FunctionType::get(ret, llvm_ref(elems), false);
        }

        case Node_Sigma: {
            // TODO watch out for cycles!
            const Sigma* sigma = type->as<Sigma>();
            Array<llvm::Type*> elems(sigma->size());
            size_t num = 0;
            for_all (elem, sigma->elems())
                elems[num++] = map(elem);
            elems.shrink(num);
            return llvm::StructType::get(context, llvm_ref(elems));
        }

        default:
            assert(!type->is_corenode());
            return hook.map(type);
    }
}

//------------------------------------------------------------------------------

void emit(World& world, EmitHook& hook) {
    CodeGen cg(world, hook);
    cg.emit();
}

//------------------------------------------------------------------------------

} // namespace anydsl2
} // namespace be_llvm
