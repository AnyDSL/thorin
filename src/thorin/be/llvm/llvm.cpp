#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/SourceMgr.h>

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
#include "thorin/util/array.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/spir.h"
#include "thorin/transform/import.h"

namespace thorin {

static llvm::Function* insert(llvm::Module* src, llvm::Module* dst, const char* name) {
    return llvm::cast<llvm::Function>(dst->getOrInsertFunction(name, src->getFunction(name)->getFunctionType()));
}

CodeGen::CodeGen(World& world, llvm::CallingConv::ID calling_convention)
    : world_(world)
    , context_()
    , module_(new llvm::Module(world.name(), context_))
    , builder_(context_)
    , calling_convention_(calling_convention)
#define NVVM_DECL(fun_name) \
    , fun_name ## _(nullptr)
#include "nvvm_decls.h"
#define SPIR_DECL(fun_name) \
    , fun_name ## _(nullptr)
#include "spir_decls.h"
{
    llvm::SMDiagnostic diag;
    AutoPtr<llvm::Module> nvvm_mod = llvm::ParseIRFile("nvvm.s", diag, context_);
    if (nvvm_mod) {
        nvvm_device_ptr_ty_ = llvm::IntegerType::getInt64Ty(context_);
#define NVVM_DECL(fun_name) \
        fun_name ## _ = insert(nvvm_mod, module_, #fun_name);
#include "nvvm_decls.h"
    }

    AutoPtr<llvm::Module> spir_mod = llvm::ParseIRFile("spir.s", diag, context_);
    if (spir_mod) {
        spir_device_ptr_ty_ = IntegerType::getInt64Ty(context_);

#define SPIR_DECL(fun_name) \
        fun_name ## _ = insert(spir_mod, module_, #fun_name);
#include "spir_decls.h"
    }
}

Lambda* CodeGen::emit_builtin(Lambda* lambda) {
    Lambda* to = lambda->to()->as_lambda();
    if (to->attribute().is(Lambda::NVVM))
        return emit_nvvm(lambda);
    else if (to->attribute().is(Lambda::SPIR))
        return emit_spir(lambda);
    THORIN_UNREACHABLE;
    return nullptr;
}

llvm::Function* CodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    llvm::FunctionType* ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, module_);
}

void CodeGen::emit() {
    // map all root-level lambdas to llvm function stubs
    for (auto lambda : top_level_lambdas(world_)) {
        if (lambda->is_builtin())
            continue;
        llvm::Function* f = nullptr;
        std::string name = lambda->name;
        if (lambda->attribute().is(Lambda::Intrinsic)) {
            std::transform(name.begin(), name.end(), name.begin(), [] (char c) { return c == '_' ? '.' : c; });
            name = "llvm." + name;
            f = emit_intrinsic_decl(name, lambda);
        } else
            f = emit_function_decl(name, lambda);

        assert(f != nullptr && "invalid function declaration");
        fcts_.emplace(lambda, f);
    }

    // emit all globals
    for (auto primop : world_.primops()) {
        if (auto global = primop->isa<Global>()) {
            llvm::Value* val;
            if (auto lambda = global->init()->isa_lambda())
                val = fcts_[lambda];
            else {
                auto var = llvm::cast<llvm::GlobalVariable>(module_->getOrInsertGlobal(global->name, map(global->referenced_type())));
                var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));
                val = var;
            }
            primops_[global] = val;
        }
    }

    // for all top-level functions
    for (auto lf : fcts_) {
        Lambda* lambda = lf.first;
        if (lambda->is_builtin() || lambda->empty())
            continue;

        assert(lambda->is_returning() || lambda->is_connected_to_builtin());
        llvm::Function* fct = lf.second;

        // map params
        const Param* ret_param = 0;
        auto arg = fct->arg_begin();
        for (auto param : lambda->params()) {
            if (param->type()->isa<Mem>())
                continue;
            if (param->order() == 0) {
                arg->setName(param->name);
                params_[param] = arg++;
            }
            else {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        Scope scope(lambda);
        BBMap bbs;

        for (auto lambda : scope.rpo()) {
            // map all bb-like lambdas to llvm bb stubs
            auto bb = bbs[lambda] = llvm::BasicBlock::Create(context_, lambda->name, fct);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && !scope.is_entry(lambda)) {
                for (auto param : lambda->params())
                    if (!param->type()->isa<Mem>())
                        phis_[param] = llvm::PHINode::Create(map(param->type()), (unsigned) param->peek().size(), param->name, bb);
            }

        }

        // never use early schedule here - this may break memory operations
        Schedule schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto lambda : scope.rpo()) {
            if (lambda->empty())
                continue;
            assert(scope.is_entry(lambda) || lambda->is_basicblock());
            builder_.SetInsertPoint(bbs[lambda]);

            for (auto primop :  schedule[lambda]) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type()->isa<Pi>() && !primop->type()->isa<Frame>()
                        && (!primop->type()->isa<Mem>() || primop->isa<Store>()))
                    primops_[primop] = emit(primop);
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: builder_.CreateRetVoid(); break;
                    case 1:
                            if (lambda->arg(0)->type()->isa<Mem>())
                                builder_.CreateRetVoid();
                            else
                                builder_.CreateRet(lookup(lambda->arg(0)));
                            break;
                    case 2: {
                                if (lambda->arg(0)->type()->isa<Mem>()) {
                                    builder_.CreateRet(lookup(lambda->arg(1)));
                                    break;
                                } else if (lambda->arg(1)->type()->isa<Mem>()) {
                                    builder_.CreateRet(lookup(lambda->arg(0)));
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
                                 llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context_, llvm_ref(elems)));

                                 for (size_t i = 0; i != n; ++i)
                                     agg = builder_.CreateInsertValue(agg, values[i], { unsigned(i) });

                                 builder_.CreateRet(agg);
                                 break;
                             }
                }
            } else if (auto select = lambda->to()->isa<Select>()) { // conditional branch
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bbs[select->tval()->as_lambda()];
                llvm::BasicBlock* fbb = bbs[select->fval()->as_lambda()];
                builder_.CreateCondBr(cond, tbb, fbb);
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())      // ordinary jump
                    builder_.CreateBr(bbs[to_lambda]);
                else {
                    if (to_lambda->is_builtin()) {
                        Lambda* ret_lambda = emit_builtin(lambda);
                        builder_.CreateBr(bbs[ret_lambda]);
                    } else {
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
                        llvm::CallInst* call = builder_.CreateCall(fcts_[to_lambda], llvm_ref(args));
                        call->setCallingConv(calling_convention_); // set proper calling convention

                        if (ret_arg == ret_param)       // call + return
                            builder_.CreateRet(call);
                        else {                          // call + continuation
                            Lambda* succ = ret_arg->as_lambda();
                            const Param* param = succ->param(0)->type()->isa<Mem>() ? nullptr : succ->param(0);
                            if (param == nullptr && succ->num_params() == 2)
                                param = succ->param(1);

                            builder_.CreateBr(bbs[succ]);
                            if (param) {
                                auto i = phis_.find(param);
                                if (i != phis_.end())
                                    i->second->addIncoming(call, builder_.GetInsertBlock());
                                else
                                    params_[param] = call;
                            }
                        }
                    }
                }
            }
        }

        // add missing arguments to phis_
        for (auto p : phis_) {
            const Param* param = p.first;
            llvm::PHINode* phi = p.second;

            for (auto peek : param->peek())
                phi->addIncoming(lookup(peek.def()), bbs[peek.from()]);
        }

        params_.clear();
        phis_.clear();
        primops_.clear();
    }

    {
        std::string error;
        auto bc_name = world_.name() + ".bc";
        llvm::raw_fd_ostream out(bc_name.c_str(), error, llvm::raw_fd_ostream::F_Binary);
        if (!error.empty())
            throw std::runtime_error("cannot write '" + bc_name + "': " + error);

        llvm::WriteBitcodeToFile(module_, out);
    }

    {
        std::string error;
        auto ll_name = world_.name() + ".ll";
        llvm::raw_fd_ostream out(ll_name.c_str(), error);
        if (!error.empty())
            throw std::runtime_error("cannot write '" + ll_name + "': " + error);

        module_->print(out, nullptr);
    }

#ifndef NDEBUG
    llvm::verifyModule(*this->module_);
#endif
}

llvm::Value* CodeGen::lookup(Def def) {
    if (def->is_const())
        return emit(def);

    if (auto primop = def->isa<PrimOp>())
        return primops_[primop];

    const Param* param = def->as<Param>();
    auto i = params_.find(param);
    if (i != params_.end())
        return i->second;

    assert(phis_.find(param) != phis_.end());
    return phis_[param];
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::Type* type, const std::string& name) {
    assert(type->isArrayTy());
    auto entry = &builder_.GetInsertBlock()->getParent()->getEntryBlock();
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

        switch (def->as<ArithOp>()->arithop_kind()) {
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

    if (auto conv = def->isa<ConvOp>()) {
        llvm::Value* from = lookup(conv->from());
        llvm::Type* to = map(conv->type());

        switch (conv->convop_kind()) {
            case ConvOp_trunc:    return builder_.CreateTrunc   (from, to);
            case ConvOp_zext:     return builder_.CreateZExt    (from, to);
            case ConvOp_sext:     return builder_.CreateSExt    (from, to);
            case ConvOp_stof:     return builder_.CreateSIToFP  (from, to);
            case ConvOp_utof:     return builder_.CreateSIToFP  (from, to);
            case ConvOp_ftrunc:   return builder_.CreateFPTrunc (from, to);
            case ConvOp_ftos:     return builder_.CreateFPToSI  (from, to);
            case ConvOp_ftou:     return builder_.CreateFPToUI  (from, to);
            case ConvOp_fext:     return builder_.CreateFPExt   (from, to);
            case ConvOp_bitcast:  return builder_.CreateBitCast (from, to);
            case ConvOp_inttoptr: return builder_.CreateIntToPtr(from, to);
            case ConvOp_ptrtoint: return builder_.CreatePtrToInt(from, to);
        }
    }

    if (auto select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
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
        llvm::Value* args[2] = { builder_.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = builder_.getInt64(i++);
            auto gep = llvm::GetElementPtrInst::CreateInBounds(alloca, args, op->name);
            gep->insertAfter(cur);
            auto store = new llvm::StoreInst(lookup(op), gep);
            store->insertAfter(gep);
            cur = store;
        }

        return builder_.CreateLoad(alloca);
    }

    if (auto tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i)
            agg = builder_.CreateInsertValue(agg, lookup(tuple->op(i)), { unsigned(i) });
        return agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto agg = lookup(aggop->agg());
        auto idx = lookup(aggop->index());

        if (aggop->agg_type()->isa<Sigma>()) {
            unsigned i = aggop->index()->primlit_value<unsigned>();

            if (aggop->isa<Extract>())
                return builder_.CreateExtractValue(agg, { i });

            auto insert = def->as<Insert>();
            auto value = lookup(insert->value());

            return builder_.CreateInsertValue(agg, value, { i });
        } else if (aggop->agg_type()->isa<ArrayType>()) {
            // TODO use llvm::ConstantArray if applicable
            std::cout << "warning: slow" << std::endl;
            auto alloca = emit_alloca(agg->getType(), aggop->name);
            builder_.CreateStore(agg, alloca);

            llvm::Value* args[2] = { builder_.getInt64(0), idx };
            auto gep = builder_.CreateInBoundsGEP(alloca, args);

            if (aggop->isa<Extract>())
                return builder_.CreateLoad(gep);

            builder_.CreateStore(lookup(aggop->as<Insert>()->value()), gep);
            return builder_.CreateLoad(alloca);
        } else {
            if (aggop->isa<Extract>())
                return builder_.CreateExtractElement(agg, idx);
            return builder_.CreateInsertElement(agg, lookup(aggop->as<Insert>()->value()), idx);
        }
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* type = map(primlit->type());
        Box box = primlit->value();

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

    if (auto undef = def->isa<Undef>()) // bottom and any
        return llvm::UndefValue::get(map(undef->type()));

    if (auto load = def->isa<Load>())
        return builder_.CreateLoad(lookup(load->ptr()));

    if (auto store = def->isa<Store>())
        return builder_.CreateStore(lookup(store->val()), lookup(store->ptr()));

    if (auto slot = def->isa<Slot>())
        return builder_.CreateAlloca(map(slot->type()->as<Ptr>()->referenced_type()), 0, slot->unique_name());

    if (def->isa<Enter>() || def->isa<Leave>())
        return nullptr;

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(map(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = builder_.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world_.literal_u32(i)));

        return vec;
    }

    if (auto lea = def->isa<LEA>()) {
        if (lea->referenced_type()->isa<Sigma>())
            return builder_.CreateConstInBoundsGEP2_64(lookup(lea->ptr()), 0ull, lea->index()->primlit_value<u64>());

        assert(lea->referenced_type()->isa<ArrayType>());
        llvm::Value* args[2] = { builder_.getInt64(0), lookup(lea->index()) };
        return builder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
    }

    THORIN_UNREACHABLE;
}

llvm::Type* CodeGen::map(const Type* type) {
    assert(!type->isa<Mem>());
    llvm::Type* llvm_type;
    switch (type->kind()) {
        case Node_PrimType_u1:  llvm_type = llvm::IntegerType::get(context_,  1); break;
        case Node_PrimType_u8:  llvm_type = llvm::IntegerType::get(context_,  8); break;
        case Node_PrimType_u16: llvm_type = llvm::IntegerType::get(context_, 16); break;
        case Node_PrimType_u32: llvm_type = llvm::IntegerType::get(context_, 32); break;
        case Node_PrimType_u64: llvm_type = llvm::IntegerType::get(context_, 64); break;
        case Node_PrimType_f32: llvm_type = llvm::Type::getFloatTy(context_);     break;
        case Node_PrimType_f64: llvm_type = llvm::Type::getDoubleTy(context_);    break;
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
                        ret = llvm::Type::getVoidTy(context_);
                    else if (pi->size() == 1)
                        ret = pi->elem(0)->isa<Mem>() ? llvm::Type::getVoidTy(context_) : map(pi->elem(0));
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
                        ret = llvm::StructType::get(context_, llvm_ref(elems));
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
            return llvm::StructType::get(context_, llvm_ref(elems));
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
    World nvvm(world.name() + "_nvvm");
    World spir(world.name() + "_spir");

    // determine different parts of the world which need to be compiled differently
    for (auto lambda : top_level_lambdas(world)) {
        if (lambda->is_connected_to_builtin(Lambda::NVVM))
            import(nvvm, lambda)->name = lambda->unique_name();
        else if (lambda->is_connected_to_builtin(Lambda::SPIR))
            import(spir, lambda)->name = lambda->unique_name();
        else
            continue;

        lambda->name = lambda->unique_name();
        lambda->destroy_body();
        lambda->attribute().set(Lambda::Extern);
    }

    if (!nvvm.lambdas().empty() || !spir.lambdas().empty())
        world.cleanup();

    CPUCodeGen(world).emit();
    if (!nvvm.lambdas().empty())
        NVVMCodeGen(nvvm).emit();
    if (!spir.lambdas().empty())
        SPIRCodeGen(spir).emit();
}

//------------------------------------------------------------------------------

} // namespace thorin
