#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "llvm/PassManager.h"
#include <llvm/Analysis/Verifier.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

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
#include "thorin/util/push.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/opencl.h"
#include "thorin/be/llvm/spir.h"
#include "thorin/transform/import.h"

#include "thorin/be/llvm/runtimes/cuda_runtime.h"
#include "thorin/be/llvm/runtimes/nvvm_runtime.h"
#include "thorin/be/llvm/runtimes/spir_runtime.h"
#include "thorin/be/llvm/runtimes/opencl_runtime.h"

namespace thorin {

CodeGen::CodeGen(World& world, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention)
    : world_(world)
    , context_()
    , module_(new llvm::Module(world.name(), context_))
    , builder_(context_)
    , function_calling_convention_(function_calling_convention)
    , device_calling_convention_(device_calling_convention)
    , kernel_calling_convention_(kernel_calling_convention)
{
    runtime_ = new GenericRuntime(context_, module_, builder_);
    cuda_runtime_ = new CUDARuntime(context_, module_, builder_);
    nvvm_runtime_ = new NVVMRuntime(context_, module_, builder_);
    spir_runtime_ = new SPIRRuntime(context_, module_, builder_);
    opencl_runtime_ = new OpenCLRuntime(context_, module_, builder_);
}

Lambda* CodeGen::emit_builtin(llvm::Function* current, Lambda* lambda) {
    Lambda* to = lambda->to()->as_lambda();
    if (to->intrinsic().is(Lambda::CUDA))
        return cuda_runtime_->emit_host_code(*this, lambda);
    if (to->intrinsic().is(Lambda::NVVM))
        return nvvm_runtime_->emit_host_code(*this, lambda);
    if (to->intrinsic().is(Lambda::SPIR))
        return spir_runtime_->emit_host_code(*this, lambda);
    if (to->intrinsic().is(Lambda::OPENCL))
        return opencl_runtime_->emit_host_code(*this, lambda);
    if (to->intrinsic().is(Lambda::Parallel))
        return runtime_->emit_parallel_start_code(*this, lambda);

    assert(to->intrinsic().is(Lambda::Vectorize));
#ifdef WFV2_SUPPORT
    return emit_vectorized(current, lambda);
#else
    assert(false && "vectorization not supported: missing WFV2");
    return nullptr;
#endif
}

llvm::Function* CodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    auto ft = llvm::cast<llvm::FunctionType>(map(lambda->type()));
    // TODO: factor emit_function_decl code
    auto fun = llvm::cast<llvm::Function>(module_->getOrInsertFunction(name, ft));
    fun->setLinkage(llvm::Function::ExternalLinkage);
    return fun;
}

void CodeGen::emit(int opt) {
    auto scopes = top_level_scopes(world_);
    // map all root-level lambdas to llvm function stubs
    for (auto scope : scopes) {
        auto lambda = scope->entry();
        if (lambda->is_builtin())
            continue;
        llvm::Function* f = nullptr;
        std::string name = lambda->unique_name();
        if (lambda->attribute().is(Lambda::Extern | Lambda::Device))
            name = lambda->name;
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

    // emit connected functions first
    std::stable_sort(scopes.begin(), scopes.end(), [] (Scope* s1, Scope* s2) { return s1->entry()->is_connected_to_builtin(); });

    for (auto ptr_scope : scopes) {
        auto& scope = *ptr_scope;
        auto lambda = scope.entry();
        if (lambda->is_builtin() || lambda->empty())
            continue;

        assert(lambda->is_returning());
        llvm::Function* fct = fcts_[lambda];

        // map params
        const Param* ret_param = nullptr;
        auto arg = fct->arg_begin();
        for (auto param : lambda->params()) {
            if (param->type().isa<MemType>())
                continue;
            if (param->order() == 0) {
                auto argv = &*arg;
                auto value = map_param(fct, argv, param);
                if (value == argv) {
                    // use param
                    arg->setName(param->unique_name());
                    params_[param] = arg++;
                } else {
                    // use provided value
                    params_[param] = value;
                }
            }
            else {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        BBMap bbs;

        for (auto lambda : scope.rpo()) {
            // map all bb-like lambdas to llvm bb stubs
            auto bb = bbs[lambda] = llvm::BasicBlock::Create(context_, lambda->name, fct);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!lambda->is_cascading() && scope.entry() != lambda) {
                for (auto param : lambda->params())
                    if (!param->type().isa<MemType>())
                        phis_[param] = llvm::PHINode::Create(map(param->type()), (unsigned) param->peek().size(), param->name, bb);
            }

        }
        auto oldStartBB = fct->begin();
        auto startBB = llvm::BasicBlock::Create(context_, fct->getName() + "_start", fct, oldStartBB);
        builder_.SetInsertPoint(startBB);
        emit_function_start(startBB, fct, lambda);
        builder_.CreateBr(oldStartBB);


        // never use early schedule here - this may break memory operations
        Schedule schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto lambda : scope.rpo()) {
            if (lambda->empty())
                continue;
            assert(lambda == scope.entry() || lambda->is_basicblock());
            builder_.SetInsertPoint(bbs[lambda]);

            for (auto primop : schedule[lambda]) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type().isa<FnType>() && !primop->type().isa<FrameType>()
                        && (!primop->type().isa<MemType>() || primop->isa<Store>()))
                    primops_[primop] = emit(primop);
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: builder_.CreateRetVoid(); break;
                    case 1:
                        if (lambda->arg(0)->type().isa<MemType>())
                            builder_.CreateRetVoid();
                        else
                            builder_.CreateRet(lookup(lambda->arg(0)));
                        break;
                    case 2: {
                        if (lambda->arg(0)->type().isa<MemType>()) {
                            builder_.CreateRet(lookup(lambda->arg(1)));
                            break;
                        } else if (lambda->arg(1)->type().isa<MemType>()) {
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
                            if (!lambda->arg(n)->type().isa<MemType>()) {
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
            } else if (lambda->to()->isa<Bottom>()) {
                builder_.CreateUnreachable();
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())         // ordinary jump
                    builder_.CreateBr(bbs[to_lambda]);
                else {
                    if (to_lambda->is_builtin()) {
                        Lambda* ret_lambda = emit_builtin(fct, lambda);
                        builder_.CreateBr(bbs[ret_lambda]);
                    } else {
                        // put all first-order args into an array
                        std::vector<llvm::Value*> args;
                        Def ret_arg;
                        for (auto arg : lambda->args()) {
                            if (arg->order() == 0) {
                                if (!arg->type().isa<MemType>())
                                    args.push_back(lookup(arg));
                            } else {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }
                        llvm::CallInst* call = builder_.CreateCall(fcts_[to_lambda], args);
                        // set proper calling convention
                        if (to_lambda->attribute().is(Lambda::KernelEntry)) {
                            call->setCallingConv(kernel_calling_convention_);
                        } else if (to_lambda->attribute().is(Lambda::Device)) {
                            call->setCallingConv(device_calling_convention_);
                        } else {
                            call->setCallingConv(function_calling_convention_);
                        }

                        if (ret_arg == ret_param) {     // call + return
                            builder_.CreateRet(call);
                        } else {                        // call + continuation
                            Lambda* succ = ret_arg->as_lambda();
                            const Param* param = succ->param(0)->type().isa<MemType>() ? nullptr : succ->param(0);
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

    // remove marked functions
    for (llvm::Function* rem : fcts_to_remove_) {
        rem->removeFromParent();
        rem->deleteBody();
    }

#ifndef NDEBUG
    llvm::verifyModule(*this->module_);
#endif
    optimize(opt);

    {
        std::string error;
        auto bc_name = get_binary_output_name(world_.name());
        llvm::raw_fd_ostream out(bc_name.c_str(), error, llvm::sys::fs::F_Binary);
        if (!error.empty())
            throw std::runtime_error("cannot write '" + bc_name + "': " + error);

        llvm::WriteBitcodeToFile(module_, out);
    }

    {
        std::string error;
        auto ll_name = get_output_name(world_.name());
        llvm::raw_fd_ostream out(ll_name.c_str(), error);
        if (!error.empty())
            throw std::runtime_error("cannot write '" + ll_name + "': " + error);

        module_->print(out, nullptr);
    }
}

void CodeGen::optimize(int opt) {
    if (opt != 0) {
        llvm::PassManagerBuilder pmbuilder;
        llvm::PassManager pass_manager;
        llvm::FunctionPassManager function_pass_manager(module_);
        if (opt == -1) {
            pmbuilder.OptLevel = 2u;
            pmbuilder.SizeLevel = 1;
        } else {
            pmbuilder.OptLevel = (unsigned) opt;
            pmbuilder.SizeLevel = 0U;
        }
        pmbuilder.DisableUnitAtATime = true;
        pmbuilder.populateFunctionPassManager(function_pass_manager);
        pmbuilder.populateModulePassManager(pass_manager);
        pass_manager.run(*module_);
    }
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
    return find(phis_, param);
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::Type* type, const std::string& name) {
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
        std::string& name = bin->name;

        if (auto cmp = bin->isa<Cmp>()) {
            auto type = cmp->lhs()->type();
            if (type->is_type_s()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return builder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return builder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return builder_.CreateICmpSGT(lhs, rhs, name);
                    case Cmp_ge: return builder_.CreateICmpSGE(lhs, rhs, name);
                    case Cmp_lt: return builder_.CreateICmpSLT(lhs, rhs, name);
                    case Cmp_le: return builder_.CreateICmpSLE(lhs, rhs, name);
                }
            } else if (type->is_type_u()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return builder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return builder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return builder_.CreateICmpUGT(lhs, rhs, name);
                    case Cmp_ge: return builder_.CreateICmpUGE(lhs, rhs, name);
                    case Cmp_lt: return builder_.CreateICmpULT(lhs, rhs, name);
                    case Cmp_le: return builder_.CreateICmpULE(lhs, rhs, name);
                }
            } else if (type->is_type_pf()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return builder_.CreateFCmpOEQ (lhs, rhs, name);
                    case Cmp_ne: return builder_.CreateFCmpONE (lhs, rhs, name);
                    case Cmp_gt: return builder_.CreateFCmpOGT (lhs, rhs, name);
                    case Cmp_ge: return builder_.CreateFCmpOGE (lhs, rhs, name);
                    case Cmp_lt: return builder_.CreateFCmpOLT (lhs, rhs, name);
                    case Cmp_le: return builder_.CreateFCmpOLE (lhs, rhs, name);
                }
            } else if (type->is_type_qf()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return builder_.CreateFCmpUEQ(lhs, rhs, name);
                    case Cmp_ne: return builder_.CreateFCmpUNE(lhs, rhs, name);
                    case Cmp_gt: return builder_.CreateFCmpUGT(lhs, rhs, name);
                    case Cmp_ge: return builder_.CreateFCmpUGE(lhs, rhs, name);
                    case Cmp_lt: return builder_.CreateFCmpULT(lhs, rhs, name);
                    case Cmp_le: return builder_.CreateFCmpULE(lhs, rhs, name);
                }
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();
            bool q = arithop->type()->is_type_q(); // quick? -> nsw/nuw/fast float

            if (type->is_type_f()) {
                switch (arithop->arithop_kind()) {
                    case ArithOp_add: return builder_.CreateFAdd(lhs, rhs, name);
                    case ArithOp_sub: return builder_.CreateFSub(lhs, rhs, name);
                    case ArithOp_mul: return builder_.CreateFMul(lhs, rhs, name);
                    case ArithOp_div: return builder_.CreateFDiv(lhs, rhs, name);
                    case ArithOp_rem: return builder_.CreateFRem(lhs, rhs, name);
                    case ArithOp_and:
                    case ArithOp_or:
                    case ArithOp_xor:
                    case ArithOp_shl:
                    case ArithOp_shr: THORIN_UNREACHABLE;
                }
            }

            if (type->is_type_s() || type->is_bool()) {
                switch (arithop->arithop_kind()) {
                    case ArithOp_add: return builder_.CreateAdd (lhs, rhs, name, false, q);
                    case ArithOp_sub: return builder_.CreateSub (lhs, rhs, name, false, q);
                    case ArithOp_mul: return builder_.CreateMul (lhs, rhs, name, false, q);
                    case ArithOp_div: return builder_.CreateSDiv(lhs, rhs, name);
                    case ArithOp_rem: return builder_.CreateSRem(lhs, rhs, name);
                    case ArithOp_and: return builder_.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return builder_.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return builder_.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return builder_.CreateShl (lhs, rhs, name, false, q);
                    case ArithOp_shr: return builder_.CreateAShr(lhs, rhs, name);
                }
            }
            if (type->is_type_u() || type->is_bool()) {
                switch (arithop->arithop_kind()) {
                    case ArithOp_add: return builder_.CreateAdd (lhs, rhs, name, q, false);
                    case ArithOp_sub: return builder_.CreateSub (lhs, rhs, name, q, false);
                    case ArithOp_mul: return builder_.CreateMul (lhs, rhs, name, q, false);
                    case ArithOp_div: return builder_.CreateUDiv(lhs, rhs, name);
                    case ArithOp_rem: return builder_.CreateURem(lhs, rhs, name);
                    case ArithOp_and: return builder_.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return builder_.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return builder_.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return builder_.CreateShl (lhs, rhs, name, q, false);
                    case ArithOp_shr: return builder_.CreateLShr(lhs, rhs, name);
                }
            }
        }
    }

    if (auto conv = def->isa<ConvOp>()) {
        auto from = lookup(conv->from());
        auto src = conv->from()->type().as<PrimType>();
        auto dst = conv->type().as<PrimType>();
        auto to = map(dst);

        if (conv->isa<Cast>()) {
            if (src.isa<PtrType>()) {
                assert(dst->is_type_i());
                return builder_.CreatePtrToInt(from, to);
            }
            if (dst.isa<PtrType>()) {
                assert(src->is_type_i());
                return builder_.CreateIntToPtr(from, to);
            }
            if (src->is_type_f() && dst->is_type_f()) {
                assert(num_bits(src->primtype_kind()) != num_bits(dst->primtype_kind()));
                return builder_.CreateFPCast(from, to);
            }
            if (src->is_type_f()) {
                if (dst->is_type_s())
                    return builder_.CreateFPToSI(from, to);
                return builder_.CreateFPToUI(from, to);
            }
            if (dst->is_type_f()) {
                if (src->is_type_s())
                    return builder_.CreateSIToFP(from, to);
                return builder_.CreateSIToFP(from, to);
            }
            if (src->is_type_i() && dst->is_type_i() && (num_bits(src->primtype_kind()) > num_bits(dst->primtype_kind())))
                return builder_.CreateTrunc(from, to);
            if (src->is_type_s() && dst->is_type_s() && (num_bits(src->primtype_kind()) < num_bits(dst->primtype_kind())))
                return builder_.CreateSExt(from, to);
            if (src->is_type_u() && dst->is_type_u() && (num_bits(src->primtype_kind()) < num_bits(dst->primtype_kind())))
                return builder_.CreateZExt(from, to);

            assert(from->getType() == to);
            return from;
        }

        if (conv->isa<Bitcast>())
            return builder_.CreateBitCast(from, to);
    }

    if (auto select = def->isa<Select>()) {
        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
    }

    if (auto array = def->isa<DefiniteArray>()) {
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

    if (auto array = def->isa<IndefiniteArray>())
        return llvm::UndefValue::get(map(array->type()));

    if (auto tuple = def->isa<Tuple>()) {
        llvm::Value* agg = llvm::UndefValue::get(map(tuple->type()));
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i)
            agg = builder_.CreateInsertValue(agg, lookup(tuple->op(i)), { unsigned(i) });
        return agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto agg = lookup(aggop->agg());
        auto idx = lookup(aggop->index());

        if (aggop->agg()->type().isa<TupleType>()) {
            unsigned i = aggop->index()->primlit_value<unsigned>();

            if (auto extract = aggop->isa<Extract>()) {
                auto agg_type = extract->agg()->type();
                if (auto agg_tuple = agg_type.isa<TupleType>()) {
                    // check for a memory-mapped extract
                    // TODO: integrate memory-mappings in a nicer way :)
                    if (agg_tuple->size() == 2 &&
                        agg_tuple->elem(0).isa<MemType>() &&
                        agg_tuple->elem(1).isa<PtrType>())
                        return lookup(extract->agg());
                }
                return builder_.CreateExtractValue(agg, { i });
            }

            auto insert = def->as<Insert>();
            auto value = lookup(insert->value());

            return builder_.CreateInsertValue(agg, value, { i });
        } else if (aggop->agg()->type().isa<ArrayType>()) {
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
            case PrimType_bool:                     return builder_. getInt1(box.get_bool());
            case PrimType_ps8:  case PrimType_qs8:  return builder_. getInt8(box. get_s8());
            case PrimType_pu8:  case PrimType_qu8:  return builder_. getInt8(box. get_u8());
            case PrimType_ps16: case PrimType_qs16: return builder_.getInt16(box.get_s16());
            case PrimType_pu16: case PrimType_qu16: return builder_.getInt16(box.get_u16());
            case PrimType_ps32: case PrimType_qs32: return builder_.getInt32(box.get_s32());
            case PrimType_pu32: case PrimType_qu32: return builder_.getInt32(box.get_u32());
            case PrimType_ps64: case PrimType_qs64: return builder_.getInt64(box.get_s64());
            case PrimType_pu64: case PrimType_qu64: return builder_.getInt64(box.get_u64());
            case PrimType_pf32: case PrimType_qf32: return llvm::ConstantFP::get(type, box.get_f32());
            case PrimType_pf64: case PrimType_qf64: return llvm::ConstantFP::get(type, box.get_f64());
        }
    }

    if (auto undef = def->isa<Undef>()) // bottom and any
        return llvm::UndefValue::get(map(undef->type()));

    if (auto alloc = def->isa<Alloc>()) { // TODO factor this code
        auto llvm_malloc = module_->getOrInsertFunction(get_alloc_name(), builder_.getInt8PtrTy(), builder_.getInt64Ty(), nullptr);
        auto alloced_type = map(alloc->alloced_type());
        llvm::CallInst* void_ptr;
        auto layout = llvm::DataLayout(module_->getDataLayout());
        if (auto array = alloc->alloced_type()->is_indefinite()) {
            auto size = builder_.CreateAdd(
                    builder_.getInt64(layout.getTypeAllocSize(alloced_type)),
                    builder_.CreateMul(builder_.CreateIntCast(lookup(alloc->extra()), builder_.getInt64Ty(), false),
                        builder_.getInt64(layout.getTypeAllocSize(map(array->elem_type())))));
            void_ptr = builder_.CreateCall(llvm_malloc, size);
        } else
            void_ptr = builder_.CreateCall(llvm_malloc, builder_.getInt64(layout.getTypeAllocSize(alloced_type)));

        auto ptr = builder_.CreatePointerCast(void_ptr, map(alloc->type()));
        return ptr;
    }

    if (auto load = def->isa<Load>())
        return emit_load(load);

    if (auto store = def->isa<Store>())
        return emit_store(store);

    if (auto slot = def->isa<Slot>())
        return builder_.CreateAlloca(map(slot->type().as<PtrType>()->referenced_type()), 0, slot->unique_name());

    if (auto map = def->isa<Map>())
        return emit_map(map);

    if (auto unmap = def->isa<Unmap>())
        return emit_unmap(unmap);

    if (def->isa<Enter>() || def->isa<Leave>())
        return nullptr;

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(map(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = builder_.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world_.literal_pu32(i)));

        return vec;
    }

    if (auto lea = def->isa<LEA>())
        return emit_lea(lea);

    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_load(Def def) {
    auto load = def->as<Load>();
    return builder_.CreateLoad(lookup(load->ptr()));
}

llvm::Value* CodeGen::emit_store(Def def) {
    auto store = def->as<Store>();
    return builder_.CreateStore(lookup(store->val()), lookup(store->ptr()));
}

llvm::Value* CodeGen::emit_lea(Def def) {
    auto lea = def->as<LEA>();
    if (lea->referenced_type().isa<TupleType>())
        return builder_.CreateConstInBoundsGEP2_64(lookup(lea->ptr()), 0ull, lea->index()->primlit_value<u64>());

    assert(lea->referenced_type().isa<ArrayType>());
    llvm::Value* args[2] = { builder_.getInt64(0), lookup(lea->index()) };
    return builder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
}

llvm::Value* CodeGen::emit_map(Def def) {
    auto map = def->as<Map>();
    // emit proper runtime call
    return runtime_->map(map->device(), (uint32_t)map->addr_space(), lookup(map->ptr()),
                         lookup(map->top_left()), lookup(map->region_size()));
}

llvm::Value* CodeGen::emit_unmap(Def def) {
    auto unmap = def->as<Unmap>();
    // emit proper runtime call
    return runtime_->unmap(unmap->device(), (uint32_t)unmap->addr_space(), lookup(unmap->ptr()));
}

// TODO factor emit_shared_map/emit_shared_unmap with the help of its base class MapOp

llvm::Value* CodeGen::emit_shared_map(Def def) {
    auto map = def->as<Map>();
    assert(map->addr_space() == AddressSpace::Shared &&
            "Only shared memory can be mapped inside NVVM code");
    auto region_size = map->region_size()->as<Tuple>();
    auto total_region_size = region_size->op(0)->as<PrimLit>()->ps32_value() *
                             region_size->op(1)->as<PrimLit>()->ps32_value() *
                             region_size->op(2)->as<PrimLit>()->ps32_value();
    // construct array type
    auto elem_type = map->ptr_type()->referenced_type().as<ArrayType>()->elem_type();
    auto type = this->map(map->world().definite_array_type(elem_type, total_region_size));
    auto global = emit_global_memory(type, map->unique_name(), 3);
    return global;
}

llvm::Value* CodeGen::emit_shared_unmap(Def def) {
    // TODO
    return nullptr;
}

llvm::Type* CodeGen::map(Type type) {
    assert(!type.isa<MemType>());
    llvm::Type* llvm_type;
    switch (type->kind()) {
        case PrimType_bool:                                                             llvm_type = builder_. getInt1Ty(); break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  llvm_type = builder_. getInt8Ty(); break;
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: llvm_type = builder_.getInt16Ty(); break;
        case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: llvm_type = builder_.getInt32Ty(); break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: llvm_type = builder_.getInt64Ty(); break;
        case PrimType_pf32: case PrimType_qf32:                                         llvm_type = builder_.getFloatTy(); break;
        case PrimType_pf64: case PrimType_qf64:                                         llvm_type = builder_.getDoubleTy();break;
        case Node_PtrType: {
            auto ptr = type.as<PtrType>();
            unsigned address_space;
            switch(ptr->addr_space())
            {
            case AddressSpace::Generic:
                address_space = 0;
                break;
            case AddressSpace::Global:
                address_space = 1;
                break;
            case AddressSpace::Texture:
                address_space = 2;
                break;
            case AddressSpace::Shared:
                address_space = 3;
                break;
            case AddressSpace::Constant:
                address_space = 4;
                break;
            default:
                THORIN_UNREACHABLE;
                break;
            }
            llvm_type = llvm::PointerType::get(map(ptr->referenced_type()), address_space);
            break;
        }
        case Node_IndefiniteArrayType:
            return llvm::ArrayType::get(map(type.as<ArrayType>()->elem_type()), 0);
        case Node_DefiniteArrayType: {
            auto array = type.as<DefiniteArrayType>();
            return llvm::ArrayType::get(map(array->elem_type()), array->dim());
        }
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type.as<FnType>();
            llvm::Type* ret = nullptr;
            std::vector<llvm::Type*> elems;
            for (auto elem : fn->elems()) {
                if (elem.isa<MemType>())
                    continue;
                if (auto fn = elem.isa<FnType>()) {
                    assert(!ret && "only one 'return' supported");
                    if (fn->empty())
                        ret = llvm::Type::getVoidTy(context_);
                    else if (fn->size() == 1)
                        ret = fn->elem(0).isa<MemType>() ? llvm::Type::getVoidTy(context_) : map(fn->elem(0));
                    else if (fn->size() == 2) {
                        if (fn->elem(0).isa<MemType>())
                            ret = map(fn->elem(1));
                        else if (fn->elem(1).isa<MemType>())
                            ret = map(fn->elem(0));
                        else
                            goto multiple;
                    } else {
multiple:
                        std::vector<llvm::Type*> elems;
                        for (auto elem : fn->elems()) {
                            if (!elem.isa<MemType>())
                                elems.push_back(map(elem));
                        }
                        ret = llvm::StructType::get(context_, elems);
                    }
                } else
                    elems.push_back(map(elem));
            }
            assert(ret);

            return llvm::FunctionType::get(ret, elems, false);
        }

        case Node_TupleType: {
            // TODO watch out for cycles!
            auto tuple = type.as<TupleType>();
            Array<llvm::Type*> elems(tuple->size());
            size_t num = 0;
            for (auto elem : tuple->elems())
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

llvm::GlobalVariable* CodeGen::emit_global_memory(llvm::Type* type, const std::string& name, unsigned addr_space) {
    return new llvm::GlobalVariable(*module_, type, false,
            llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(type), name,
            nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
}

//------------------------------------------------------------------------------

void emit_llvm(World& world, int opt) {
    World cuda(world.name());
    World nvvm(world.name());
    World spir(world.name());
    World opencl(world.name());

    // determine different parts of the world which need to be compiled differently
    for (auto scope : top_level_scopes(world)) {
        auto lambda = scope->entry();
        Lambda* imported = nullptr;
        if (lambda->is_connected_to_builtin(Lambda::CUDA))
            imported = import(cuda, lambda)->as_lambda();
        else if (lambda->is_connected_to_builtin(Lambda::NVVM))
            imported = import(nvvm, lambda)->as_lambda();
        else if (lambda->is_connected_to_builtin(Lambda::SPIR))
            imported = import(spir, lambda)->as_lambda();
        else if (lambda->is_connected_to_builtin(Lambda::OPENCL))
            imported = import(opencl, lambda)->as_lambda();
        else
            continue;

        imported->name = lambda->unique_name();
        imported->attribute().set(Lambda::Extern | Lambda::KernelEntry);
        lambda->name = lambda->unique_name();
        lambda->destroy_body();
        lambda->attribute().set(Lambda::Extern);

        for (size_t i = 0, e = lambda->num_params(); i != e; ++i)
            imported->param(i)->name = lambda->param(i)->unique_name();
    }

    if (!cuda.lambdas().empty() || !nvvm.lambdas().empty() || !spir.lambdas().empty() || !opencl.lambdas().empty())
        world.cleanup();

    CPUCodeGen(world).emit(opt);
    if (!cuda.  lambdas().empty()) CUDACodeGen(cuda).emit(/*opt*/);
    if (!nvvm.  lambdas().empty()) NVVMCodeGen(nvvm).emit(opt);
    if (!spir.  lambdas().empty()) SPIRCodeGen(spir).emit(opt);
    if (!opencl.lambdas().empty()) OpenCLCodeGen(opencl).emit(/*opt*/);
}

//------------------------------------------------------------------------------

}
