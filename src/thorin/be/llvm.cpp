#ifdef LLVM_SUPPORT

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/PassManager.h>
#include <llvm/Transforms/Scalar.h>

#ifdef WFV2_SUPPORT
#include <wfvInterface.h>
#endif

#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

template<class T> 
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

//------------------------------------------------------------------------------

typedef LambdaMap<llvm::BasicBlock*> BBMap;

class CodeGen {
public:
    CodeGen(World& world)
        : world(world)
        , context()
        , builder(context)
        , module(new llvm::Module(world.name(), context))
    {}

    void emit();
    llvm::Type* map(const Type* type);
    llvm::Value* emit(Def def);
    llvm::Value* lookup(Def def);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string& name);

private:
    World& world;
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    AutoPtr<llvm::Module> module;
    std::unordered_map<const Param*, llvm::Value*> params;
    std::unordered_map<const Param*, llvm::PHINode*> phis;
    std::unordered_map<const PrimOp*, llvm::Value*> primops;
    std::unordered_map<Lambda*, llvm::Function*> fcts;
};

//------------------------------------------------------------------------------

void CodeGen::emit() {
    std::unordered_map<Lambda*, const Param*> ret_map;
    // map all root-level lambdas to llvm function stubs
    for (auto lambda : top_level_lambdas(world)) {
        if (lambda->is_builtin())
            continue;
        llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
        std::string name = lambda->name;
        if (lambda->attribute().is(Lambda::Intrinsic)) {
            std::transform(name.begin(), name.end(), name.begin(), [] (char c) { return c == '_' ? '.' : c; });
            name = "llvm." + name;
        }
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, module);
        fcts.emplace(lambda, f);
    }

    // emit all globals
    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>()) {
            auto val = llvm::cast<llvm::GlobalValue>(module->getOrInsertGlobal(global->name, map(global->referenced_type())));
            if (auto var = llvm::dyn_cast<llvm::GlobalVariable>(val))
                var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));
            else
                assert(global->init()->isa_lambda());
            primops[global] = val;
        }
    }

    // for all top-level functions
    for (auto lf : fcts) {
        Lambda* lambda = lf.first;
        if (lambda->is_builtin() || lambda->empty())
            continue;

        assert(lambda->is_returning() || lambda->is_connected_to_builtin());
        llvm::Function* fct = lf.second;

        // map params
        const Param* ret_param = 0;
        if (lambda->is_connected_to_builtin())
            ret_param = ret_map[lambda];
        else {
            auto arg = fct->arg_begin();
            for (auto param : lambda->params()) {
                if (param->type()->isa<Mem>())
                    continue;
                if (param->order() == 0) {
                    arg->setName(param->name);
                    params[param] = arg++;
                }
                else {
                    assert(!ret_param);
                    ret_param = param;
                }
            }
        }
        assert(ret_param);

        Scope scope(lambda);
        BBMap bbs;

        for (auto lambda : scope.rpo()) {
            // map all bb-like lambdas to llvm bb stubs
            auto bb = bbs[lambda] = llvm::BasicBlock::Create(context, lambda->name, fct);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && !scope.is_entry(lambda)) {
                for (auto param : lambda->params())
                    if (!param->type()->isa<Mem>())
                        phis[param] = llvm::PHINode::Create(map(param->type()), (unsigned) param->peek().size(), param->name, bb);
            }

        }

        // never use early schedule here - this may break memory operations
        Schedule schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto lambda : scope.rpo()) {
            if (lambda->empty())
                continue;
            assert(scope.is_entry(lambda) || lambda->is_basicblock());
            builder.SetInsertPoint(bbs[lambda]);

            for (auto primop :  schedule[lambda]) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type()->isa<Pi>() && !primop->type()->isa<Frame>()
                        && (!primop->type()->isa<Mem>() || primop->isa<Store>()))
                    primops[primop] = emit(primop);
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: builder.CreateRetVoid(); break;
                    case 1:
                        if (lambda->arg(0)->type()->isa<Mem>())
                            builder.CreateRetVoid();
                        else
                            builder.CreateRet(lookup(lambda->arg(0)));
                        break;
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

                        for (size_t i = 0; i != n; ++i)
                            agg = builder.CreateInsertValue(agg, values[i], { unsigned(i) });

                        builder.CreateRet(agg);
                        break;
                    }
                }
            } else if (auto select = lambda->to()->isa<Select>()) { // conditional branch
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bbs[select->tval()->as_lambda()];
                llvm::BasicBlock* fbb = bbs[select->fval()->as_lambda()];
                builder.CreateCondBr(cond, tbb, fbb);
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())      // ordinary jump
                    builder.CreateBr(bbs[to_lambda]);
                else {
                    // put all first-order args into an array
                    Array<llvm::Value*> args(lambda->args().size() - 1);
                    size_t i = 0;
                    Def ret_arg = 0;
                    for (auto arg : lambda->args()) {
                        if (arg->order() == 0) {
                            if (!arg->type()->isa<Mem>())
                                args[i++] = lookup(arg);
                        }
                        else {
                            assert(!ret_arg);
                            ret_arg = arg;
                        }
                    }
                    args.shrink(i);
                    llvm::CallInst* call = builder.CreateCall(fcts[to_lambda], llvm_ref(args));

                    if (ret_arg == ret_param)       // call + return
                        builder.CreateRet(call);
                    else {                          // call + continuation
                        Lambda* succ = ret_arg->as_lambda();
                        const Param* param = succ->param(0)->type()->isa<Mem>() ? nullptr : succ->param(0);
                        if (param == nullptr && succ->num_params() == 2)
                            param = succ->param(1);

                        builder.CreateBr(bbs[succ]);
                        if (param) {
                            auto i = phis.find(param);
                            if (i != phis.end())
                                i->second->addIncoming(call, builder.GetInsertBlock());
                            else
                                params[param] = call;
                        }
                    }
                }
            }
        }

        // add missing arguments to phis
        for (auto p : phis) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for (auto peek : param->peek())
                phi->addIncoming(lookup(peek.def()), bbs[peek.from()]);
        }

        params.clear();
        phis.clear();
        primops.clear();
    }

    module->dump();
#ifndef NDEBUG
    llvm::verifyModule(*this->module);
#endif
}

llvm::Value* CodeGen::lookup(Def def) {
    if (def->is_const())
        return emit(def);

    if (auto primop = def->isa<PrimOp>())
        return primops[primop];

    const Param* param = def->as<Param>();
    auto i = params.find(param);
    if (i != params.end())
        return i->second;

    assert(phis.find(param) != phis.end());
    return phis[param];
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::Type* type, const std::string& name) {
    assert(type->isArrayTy());
    auto entry = &builder.GetInsertBlock()->getParent()->getEntryBlock();
    llvm::AllocaInst* alloca;
    if (entry->empty())
        alloca = new llvm::AllocaInst(type, nullptr, name, entry);
    else
        alloca = new llvm::AllocaInst(type, nullptr, name, entry->getFirstNonPHIOrDbg());
    return alloca;
}

llvm::Value* CodeGen::emit(Def def) {
    if (auto bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());

        if (auto rel = def->isa<RelOp>()) {
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

    if (auto conv = def->isa<ConvOp>()) {
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

    if (auto select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder.CreateSelect(cond, tval, fval);
    }

    if (auto array = def->isa<ArrayAgg>()) {
        auto type = llvm::cast<llvm::ArrayType>(map(array->type()));
        if (array->is_const()) {
            size_t size = array->size();
            Array<llvm::Constant*> vals(size);
            for (size_t i = 0; i != size; ++i)
                vals[i] = llvm::cast<llvm::Constant>(emit(array->op(i)));
            return llvm::ConstantArray::get(type, llvm_ref(vals));
        }
        std::cout << "warning: slow" << std::endl;
        auto alloca = emit_alloca(type, array->name);
        llvm::Instruction* cur = alloca;

        u64 i = 0;
        llvm::Value* args[2] = { builder.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = builder.getInt64(i++);
            auto gep = llvm::GetElementPtrInst::CreateInBounds(alloca, args, op->name);
            gep->insertAfter(cur);
            auto store = new llvm::StoreInst(lookup(op), gep);
            store->insertAfter(gep);
            cur = store;
        }

        return builder.CreateLoad(alloca);
    }

    if (auto tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i)
            agg = builder.CreateInsertValue(agg, lookup(tuple->op(i)), { unsigned(i) });
        return agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto agg = lookup(aggop->agg());
        auto idx = lookup(aggop->index());

        if (aggop->agg_type()->isa<Sigma>()) {
            unsigned i = aggop->index()->primlit_value<unsigned>();

            if (auto extract = aggop->as<Extract>())
                return builder.CreateExtractValue(agg, { i });

            auto insert = def->as<Insert>();
            auto value = lookup(insert->value());

            return builder.CreateInsertValue(agg, value, { i });
        } else if (aggop->agg_type()->isa<ArrayType>()) {
            // TODO use llvm::ConstantArray if applicable
            std::cout << "warning: slow" << std::endl;
            auto alloca = emit_alloca(agg->getType(), aggop->name);
            builder.CreateStore(agg, alloca);

            llvm::Value* args[2] = { builder.getInt64(0), idx };
            auto gep = builder.CreateInBoundsGEP(alloca, args);

            if (aggop->isa<Extract>())
                return builder.CreateLoad(gep);

            builder.CreateStore(lookup(aggop->as<Insert>()->value()), gep);
            return builder.CreateLoad(alloca);
        } else {
            if (auto extract = aggop->as<Extract>())
                return builder.CreateExtractElement(agg, idx);
            return builder.CreateInsertElement(agg, lookup(aggop->as<Insert>()->value()), idx);
        }
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* type = map(primlit->type());
        Box box = primlit->value();

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

    if (auto undef = def->isa<Undef>()) // bottom and any
        return llvm::UndefValue::get(map(undef->type()));

    if (auto load = def->isa<Load>())
        return builder.CreateLoad(lookup(load->ptr()));

    if (auto store = def->isa<Store>())
        return builder.CreateStore(lookup(store->val()), lookup(store->ptr()));

    if (auto slot = def->isa<Slot>())
        return builder.CreateAlloca(map(slot->type()->as<Ptr>()->referenced_type()), 0, slot->unique_name());

    if (def->isa<Enter>() || def->isa<Leave>())
        return nullptr;

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(map(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = builder.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world.literal_u32(i)));

        return vec;
    }

    if (auto lea = def->isa<LEA>()) {
        if (lea->referenced_type()->isa<Sigma>())
            return builder.CreateConstInBoundsGEP2_64(lookup(lea->ptr()), 0ull, lea->index()->primlit_value<u64>());

        assert(lea->referenced_type()->isa<ArrayType>());
        llvm::Value* args[2] = { builder.getInt64(0), lookup(lea->index()) };
        return builder.CreateInBoundsGEP(lookup(lea->ptr()), args);
    }

    THORIN_UNREACHABLE;
}

llvm::Type* CodeGen::map(const Type* type) {
    assert(!type->isa<Mem>());
    llvm::Type* llvm_type;
    switch (type->kind()) {
        case Node_PrimType_u1:  llvm_type = llvm::IntegerType::get(context,  1); break;
        case Node_PrimType_u8:  llvm_type = llvm::IntegerType::get(context,  8); break;
        case Node_PrimType_u16: llvm_type = llvm::IntegerType::get(context, 16); break;
        case Node_PrimType_u32: llvm_type = llvm::IntegerType::get(context, 32); break;
        case Node_PrimType_u64: llvm_type = llvm::IntegerType::get(context, 64); break;
        case Node_PrimType_f32: llvm_type = llvm::Type::getFloatTy(context);     break;
        case Node_PrimType_f64: llvm_type = llvm::Type::getDoubleTy(context);    break;
        case Node_Ptr:          llvm_type = llvm::PointerType::getUnqual(map(type->as<Ptr>()->referenced_type())); break;
        case Node_IndefArray:   return llvm::ArrayType::get(map(type->as<ArrayType>()->elem_type()), 0);
        case Node_DefArray: {
            auto array = type->as<DefArray>();
            return llvm::ArrayType::get(map(array->elem_type()), array->dim());
        }
        case Node_Pi: {
            // extract "return" type, collect all other types
            const Pi* pi = type->as<Pi>();
            llvm::Type* ret = 0;
            size_t i = 0;
            Array<llvm::Type*> elems(pi->size() - 1);
            for (auto elem : pi->elems()) {
                if (elem->isa<Mem>())
                    continue;
                if (auto pi = elem->isa<Pi>()) {
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
                        size_t num = 0;
                        for (size_t j = 0, e = elems.size(); j != e; ++j) {
                            if (pi->elem(j)->isa<Mem>())
                                continue;
                            ++num;
                            elems[j] = map(pi->elem(j));
                        }
                        elems.shrink(num);
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
            for (auto elem : sigma->elems())
                elems[num++] = map(elem);
            elems.shrink(num);
            return llvm::StructType::get(context, llvm_ref(elems));
        }

        default: 
            THORIN_UNREACHABLE;
    }

    if (type->length() == 1)
        return llvm_type;
    return llvm::VectorType::get(llvm_type, type->length());
}

//------------------------------------------------------------------------------

void emit_llvm(World& world) {
    CodeGen(world).emit();
}

//------------------------------------------------------------------------------

} // namespace thorin

#endif
