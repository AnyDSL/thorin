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
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/array.h"
#include "thorin/util/push.h"
#include "thorin/analyses/bb_schedule.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
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
    , runtime_(new GenericRuntime(context_, module_, builder_))
    , cuda_runtime_(new CUDARuntime(context_, module_, builder_))
    , nvvm_runtime_(new NVVMRuntime(context_, module_, builder_))
    , spir_runtime_(new SPIRRuntime(context_, module_, builder_))
    , opencl_runtime_(new OpenCLRuntime(context_, module_, builder_))
{}

Lambda* CodeGen::emit_intrinsic(Lambda* lambda) {
    Lambda* to = lambda->to()->as_lambda();
    switch (to->intrinsic()) {
        case Intrinsic::Atomic:    return emit_atomic(lambda);
        case Intrinsic::Select4:
        case Intrinsic::Select8:
        case Intrinsic::Select16:  return emit_select(lambda);
        case Intrinsic::Shuffle4:
        case Intrinsic::Shuffle8:
        case Intrinsic::Shuffle16: return emit_shuffle(lambda);
        case Intrinsic::Munmap:    runtime_->munmap(lookup(lambda->arg(1)));
                                   return lambda->args().back()->as_lambda();
        case Intrinsic::CUDA:      return cuda_runtime_->emit_host_code(*this, lambda);
        case Intrinsic::NVVM:      return nvvm_runtime_->emit_host_code(*this, lambda);
        case Intrinsic::SPIR:      return spir_runtime_->emit_host_code(*this, lambda);
        case Intrinsic::OpenCL:    return opencl_runtime_->emit_host_code(*this, lambda);
        case Intrinsic::Parallel:  return emit_parallel(lambda);
        case Intrinsic::Spawn:     return emit_spawn(lambda);
        case Intrinsic::Sync:      return emit_sync(lambda);
#ifdef WFV2_SUPPORT
        case Intrinsic::Vectorize: return emit_vectorize_continuation(lambda);
#endif
        default: THORIN_UNREACHABLE;
    }
}

Lambda* CodeGen::emit_atomic(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    // atomic kind: Xchg Add Sub And Nand Or Xor Max Min
    u32 kind = lambda->arg(1)->as<PrimLit>()->qu32_value();
    auto ptr = lookup(lambda->arg(2));
    auto val = lookup(lambda->arg(3));
    assert(kind >= llvm::AtomicRMWInst::BinOp::Xchg && kind <= llvm::AtomicRMWInst::BinOp::UMin && "unsupported atomic");
    llvm::AtomicRMWInst::BinOp binop = (llvm::AtomicRMWInst::BinOp)kind;

    auto cont = lambda->arg(4)->as_lambda();
    params_[cont->param(1)] = builder_.CreateAtomicRMW(binop, ptr, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SynchronizationScope::CrossThread);
    return cont;
}

Lambda* CodeGen::emit_select(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    auto cond = lookup(lambda->arg(1));
    auto a = lookup(lambda->arg(2));
    auto b = lookup(lambda->arg(3));

    auto cont = lambda->arg(4)->as_lambda();
    params_[cont->param(1)] = builder_.CreateSelect(cond, a, b);
    return cont;
}

Lambda* CodeGen::emit_shuffle(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    auto mask = lookup(lambda->arg(3));
    auto a = lookup(lambda->arg(1));
    auto b = lookup(lambda->arg(2));

    auto cont = lambda->arg(4)->as_lambda();
    params_[cont->param(1)] = builder_.CreateShuffleVector(a, b, mask);
    return cont;
}

llvm::FunctionType* CodeGen::convert_fn_type(Lambda* lambda) {
    return llvm::cast<llvm::FunctionType>(convert(lambda->type()));
}

llvm::Function* CodeGen::emit_function_decl(Lambda* lambda) {
    if (auto f = find(fcts_, lambda))
        return f;

    std::string name = (lambda->is_external() || lambda->empty()) ? lambda->name : lambda->unique_name();
    auto f = llvm::cast<llvm::Function>(module_->getOrInsertFunction(name, convert_fn_type(lambda)));

    // set linkage
    if (lambda->is_external() || lambda->empty())
        f->setLinkage(llvm::Function::ExternalLinkage);
    else
        f->setLinkage(llvm::Function::InternalLinkage);

    // set calling convention
    if (lambda->is_external()) {
        f->setCallingConv(kernel_calling_convention_);
        emit_function_decl_hook(lambda, f);
    } else {
        if (lambda->cc() == CC::Device)
            f->setCallingConv(device_calling_convention_);
        else
            f->setCallingConv(function_calling_convention_);
    }

    return fcts_[lambda] = f;
}

void CodeGen::emit(int opt) {
    Scope::for_each(world_, [&] (const Scope& scope) {
        entry_ = scope.entry();
        assert(entry_->is_returning());
        llvm::Function* fct = emit_function_decl(entry_);

        // map params
        const Param* ret_param = nullptr;
        auto arg = fct->arg_begin();
        for (auto param : entry_->params()) {
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

        BBMap bb2lambda;
        auto bbs = bb_schedule(scope);

        for (auto bb_lambda : bbs) {
            // map all bb-like lambdas to llvm bb stubs
            auto bb = bb2lambda[bb_lambda] = llvm::BasicBlock::Create(context_, bb_lambda->name, fct);

            // create phi node stubs (for all non-cascading lambdas different from entry)
            if (!bb_lambda->is_cascading() && entry_ != bb_lambda) {
                for (auto param : bb_lambda->params())
                    if (!param->type().isa<MemType>())
                        phis_[param] = llvm::PHINode::Create(convert(param->type()), (unsigned) param->peek().size(), param->name, bb);
            }
        }

        auto oldStartBB = fct->begin();
        auto startBB = llvm::BasicBlock::Create(context_, fct->getName() + "_start", fct, oldStartBB);
        builder_.SetInsertPoint(startBB);
        emit_function_start(startBB, entry_);
        builder_.CreateBr(oldStartBB);
        auto schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto bb_lambda : bbs) {
            if (bb_lambda->empty())
                continue;
            assert(bb_lambda == entry_ || bb_lambda->is_basicblock());
            builder_.SetInsertPoint(bb2lambda[bb_lambda]);

            for (auto primop : schedule[bb_lambda])
                    primops_[primop] = emit(primop);

            // terminate bb
            if (bb_lambda->to() == ret_param) { // return
                size_t num_args = bb_lambda->num_args();
                switch (num_args) {
                    case 0: builder_.CreateRetVoid(); break;
                    case 1:
                        if (bb_lambda->arg(0)->type().isa<MemType>())
                            builder_.CreateRetVoid();
                        else
                            builder_.CreateRet(lookup(bb_lambda->arg(0)));
                        break;
                    case 2:
                        if (bb_lambda->arg(0)->type().isa<MemType>()) {
                            builder_.CreateRet(lookup(bb_lambda->arg(1)));
                            break;
                        } else if (bb_lambda->arg(1)->type().isa<MemType>()) {
                            builder_.CreateRet(lookup(bb_lambda->arg(0)));
                            break;
                        }
                        // FALLTHROUGH
                    default: {
                        Array<llvm::Value*> values(num_args);
                        Array<llvm::Type*> args(num_args);

                        size_t n = 0;
                        for (size_t a = 0; a < num_args; ++a) {
                            if (!bb_lambda->arg(n)->type().isa<MemType>()) {
                                llvm::Value* val = lookup(bb_lambda->arg(a));
                                values[n] = val;
                                args[n++] = val->getType();
                            }
                        }

                        assert(n == num_args || n+1 == num_args);
                        values.shrink(n);
                        args.shrink(n);
                        llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context_, llvm_ref(args)));

                        for (size_t i = 0; i != n; ++i)
                            agg = builder_.CreateInsertValue(agg, values[i], { unsigned(i) });

                        builder_.CreateRet(agg);
                        break;
                    }
                }
            } else if (auto select = bb_lambda->to()->isa<Select>()) { // conditional branch
                llvm::Value* cond = lookup(select->cond());
                llvm::BasicBlock* tbb = bb2lambda[select->tval()->as_lambda()];
                llvm::BasicBlock* fbb = bb2lambda[select->fval()->as_lambda()];
                builder_.CreateCondBr(cond, tbb, fbb);
            } else if (bb_lambda->to()->isa<Bottom>()) {
                builder_.CreateUnreachable();
            } else {
                Lambda* to_lambda = bb_lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())         // ordinary jump
                    builder_.CreateBr(bb2lambda[to_lambda]);
                else {
                    if (to_lambda->is_intrinsic()) {
                        Lambda* ret_lambda = emit_intrinsic(bb_lambda);
                        builder_.CreateBr(bb2lambda[ret_lambda]);
                    } else {
                        // put all first-order args into an array
                        std::vector<llvm::Value*> args;
                        Def ret_arg;
                        for (auto arg : bb_lambda->args()) {
                            if (arg->order() == 0) {
                                if (!arg->type().isa<MemType>())
                                    args.push_back(lookup(arg));
                            } else {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }
                        llvm::CallInst* call = builder_.CreateCall(emit_function_decl(to_lambda), args);
                        // set proper calling convention
                        if (to_lambda->is_external()) {
                            call->setCallingConv(kernel_calling_convention_);
                        } else if (to_lambda->cc() == CC::Device) {
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

                            builder_.CreateBr(bb2lambda[succ]);
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
                phi->addIncoming(lookup(peek.def()), bb2lambda[peek.from()]);
        }

        params_.clear();
        phis_.clear();
        primops_.clear();
    });

#ifdef WFV2_SUPPORT
    // emit vectorized code
    for (const auto& tuple : wfv_todo_)
        emit_vectorize(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
    wfv_todo_.clear();
#endif

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
    if (auto primop = def->isa<PrimOp>()) {
        if (auto res = find(primops_, primop))
            return res;
        else
            return primops_[primop] = emit(def);
    }

    if (auto param = def->isa<Param>()) {
        auto i = params_.find(param);
        if (i != params_.end())
            return i->second;

        assert(phis_.find(param) != phis_.end());
        return find(phis_, param);
    }

    THORIN_UNREACHABLE;
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
            } else if (type->is_type_u() || type->is_bool()) {
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
            } else if (type.isa<PtrType>()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return builder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return builder_.CreateICmpNE (lhs, rhs, name);
                    default: THORIN_UNREACHABLE;
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
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();
        auto to = convert(dst_type);

        if (from->getType() == to)
            return from;

        if (conv->isa<Cast>()) {
            if (src_type.isa<PtrType>()) {
                assert(dst_type->is_type_i() || dst_type->is_bool());
                return builder_.CreatePtrToInt(from, to);
            }
            if (dst_type.isa<PtrType>()) {
                assert(src_type->is_type_i() || dst_type->is_bool());
                return builder_.CreateIntToPtr(from, to);
            }

            auto src = src_type.as<PrimType>();
            auto dst = dst_type.as<PrimType>();

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
            if (       (src->is_type_i() || src->is_bool())
                    && (dst->is_type_i() || dst->is_bool())
                    && (num_bits(src->primtype_kind()) > num_bits(dst->primtype_kind())))
                return builder_.CreateTrunc(from, to);
            if (       (src->is_type_i() || src->is_bool())
                    && (dst->is_type_s() || dst->is_bool())
                    && (num_bits(src->primtype_kind()) < num_bits(dst->primtype_kind())))
                return builder_.CreateSExt(from, to);
            if (       (src->is_type_i() || src->is_bool())
                    && (dst->is_type_u() || dst->is_bool())
                    && (num_bits(src->primtype_kind()) < num_bits(dst->primtype_kind())))
                return builder_.CreateZExt(from, to);

            assert(false && "unsupported cast");
        }

        if (conv->isa<Bitcast>())
            return builder_.CreateBitCast(from, to);
    }

    if (auto select = def->isa<Select>()) {
        if (def->type().isa<FnType>())
            return nullptr;

        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return builder_.CreateSelect(cond, tval, fval);
    }

    if (auto array = def->isa<DefiniteArray>()) {
        auto type = llvm::cast<llvm::ArrayType>(convert(array->type()));
        if (array->is_const()) {
            size_t size = array->size();
            Array<llvm::Constant*> vals(size);
            for (size_t i = 0; i != size; ++i)
                vals[i] = llvm::cast<llvm::Constant>(emit(array->op(i)));
            return llvm::ConstantArray::get(type, llvm_ref(vals));
        }
        std::cout << "warning: slow" << std::endl;
        auto alloca = emit_alloca(type, array->name);

        u64 i = 0;
        llvm::Value* args[2] = { builder_.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = builder_.getInt64(i++);
            auto gep = builder_.CreateInBoundsGEP(alloca, args, op->name);
            builder_.CreateStore(lookup(op), gep);
        }

        return builder_.CreateLoad(alloca);
    }

    if (auto array = def->isa<IndefiniteArray>())
        return llvm::UndefValue::get(convert(array->type()));

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>());
        llvm::Value* llvm_agg = llvm::UndefValue::get(convert(agg->type()));
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i)
            llvm_agg = builder_.CreateInsertValue(llvm_agg, lookup(agg->op(i)), { unsigned(i) });
        return llvm_agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto llvm_agg = lookup(aggop->agg());
        auto llvm_idx = lookup(aggop->index());
        auto copy_to_alloca = [&] () {
            std::cout << "warning: slow" << std::endl;
            auto alloca = emit_alloca(llvm_agg->getType(), aggop->name);
            builder_.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { builder_.getInt64(0), llvm_idx };
            auto gep = builder_.CreateInBoundsGEP(alloca, args);
            return std::make_pair(alloca, gep);
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (auto memop = extract->agg()->isa<MemOp>())
                return lookup(memop);

            if (aggop->agg()->type().isa<ArrayType>())
                return builder_.CreateLoad(copy_to_alloca().second);

            if (extract->agg()->type().isa<VectorType>())
                return builder_.CreateExtractElement(llvm_agg, llvm_idx);
            // tuple/struct
            return builder_.CreateExtractValue(llvm_agg, {aggop->index()->primlit_value<unsigned>()});
        }

        auto insert = def->as<Insert>();
        auto value = lookup(insert->value());

        if (insert->agg()->type().isa<ArrayType>()) {
            auto p = copy_to_alloca();
            builder_.CreateStore(lookup(aggop->as<Insert>()->value()), p.second);
            return builder_.CreateLoad(p.first);
        }
        if (insert->agg()->type().isa<VectorType>())
            return builder_.CreateInsertElement(llvm_agg, lookup(aggop->as<Insert>()->value()), llvm_idx);
        // tuple/struct
        return builder_.CreateInsertValue(llvm_agg, value, {aggop->index()->primlit_value<unsigned>()});
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* llvm_type = convert(primlit->type());
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
            case PrimType_pf32: case PrimType_qf32: return llvm::ConstantFP::get(llvm_type, box.get_f32());
            case PrimType_pf64: case PrimType_qf64: return llvm::ConstantFP::get(llvm_type, box.get_f64());
        }
    }

    if (auto bottom = def->isa<Bottom>())
        return llvm::UndefValue::get(convert(bottom->type()));

    if (auto alloc = def->isa<Alloc>()) { // TODO factor this code
        // TODO do this only once
        auto llvm_malloc = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
                    get_alloc_name(), builder_.getInt8PtrTy(), builder_.getInt64Ty(), nullptr));
        llvm_malloc->addAttribute(llvm::AttributeSet::ReturnIndex, llvm::Attribute::NoAlias);
        auto alloced_type = convert(alloc->alloced_type());
        llvm::CallInst* void_ptr;
        auto layout = llvm::DataLayout(module_->getDataLayout());
        if (auto array = alloc->alloced_type()->is_indefinite()) {
            auto size = builder_.CreateAdd(
                    builder_.getInt64(layout.getTypeAllocSize(alloced_type)),
                    builder_.CreateMul(builder_.CreateIntCast(lookup(alloc->extra()), builder_.getInt64Ty(), false),
                        builder_.getInt64(layout.getTypeAllocSize(convert(array->elem_type())))));
            void_ptr = builder_.CreateCall(llvm_malloc, size);
        } else
            void_ptr = builder_.CreateCall(llvm_malloc, builder_.getInt64(layout.getTypeAllocSize(alloced_type)));

        return builder_.CreatePointerCast(void_ptr, convert(alloc->out_ptr_type()));
    }

    if (auto load = def->isa<Load>())    return emit_load(load);
    if (auto store = def->isa<Store>())  return emit_store(store);
    if (auto mmap = def->isa<Map>())     return emit_mmap(mmap);
    if (auto lea = def->isa<LEA>())      return emit_lea(lea);
    if (def->isa<Enter>())               return nullptr;

    if (auto slot = def->isa<Slot>())
        return builder_.CreateAlloca(convert(slot->type().as<PtrType>()->referenced_type()), 0, slot->unique_name());

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(convert(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = builder_.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world_.literal_pu32(i)));

        return vec;
    }

    if (auto global = def->isa<Global>()) {
        llvm::Value* val;
        if (auto lambda = global->init()->isa_lambda())
            val = fcts_[lambda];
        else {
            auto llvm_type = convert(global->alloced_type());
            auto var = llvm::cast<llvm::GlobalVariable>(module_->getOrInsertGlobal(global->name, llvm_type));
            if (global->init()->isa<Bottom>())
                var->setInitializer(llvm::Constant::getNullValue(llvm_type)); // HACK
            else
                var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));
            val = var;
        }
        return val;
    }

    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_load(Def def) {
    return builder_.CreateLoad(lookup(def->as<Load>()->ptr()));
}

llvm::Value* CodeGen::emit_store(Def def) {
    auto store = def->as<Store>();
    return builder_.CreateStore(lookup(store->val()), lookup(store->ptr()));
}

llvm::Value* CodeGen::emit_lea(Def def) {
    auto lea = def->as<LEA>();
    if (lea->ptr_referenced_type().isa<TupleType>() || lea->ptr_referenced_type().isa<StructAppType>())
        return builder_.CreateStructGEP(lookup(lea->ptr()), lea->index()->primlit_value<u32>());

    assert(lea->ptr_referenced_type().isa<ArrayType>());
    llvm::Value* args[2] = { builder_.getInt64(0), lookup(lea->index()) };
    return builder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
}

llvm::Value* CodeGen::emit_mmap(Def def) {
    auto mmap = def->as<Map>();
    // emit proper runtime call
    auto ref_ty = mmap->out_ptr_type()->referenced_type();
    Type type;
    if (auto array = ref_ty->is_indefinite())
        type = array->elem_type();
    else
        type = mmap->out_ptr_type()->referenced_type();
    auto layout = llvm::DataLayout(module_->getDataLayout());
    auto size = builder_.getInt32(layout.getTypeAllocSize(convert(type)));
    return runtime_->mmap(mmap->device(), (uint32_t)mmap->addr_space(), lookup(mmap->ptr()),
                          lookup(mmap->mem_offset()), lookup(mmap->mem_size()), size);
}

llvm::Value* CodeGen::emit_shared_mmap(Def def, bool prefix) {
    auto mmap = def->as<Map>();
    assert(entry_ && "shared memory can only be mapped inside kernel");
    assert(mmap->addr_space() == AddressSpace::Shared && "wrong address space for shared memory");
    auto num_elems = mmap->mem_size()->as<PrimLit>()->ps32_value();

    // construct array type
    auto elem_type = mmap->out_ptr_type()->referenced_type().as<ArrayType>()->elem_type();
    auto type = this->convert(mmap->world().definite_array_type(elem_type, num_elems));
    auto global = emit_global_memory(type, (prefix ? entry_->name + "." : "") + mmap->unique_name(), 3);
    return global;
}

llvm::Type* CodeGen::convert(Type type) {
    if (auto ltype = thorin::find(types_, *type.unify()))
        return ltype;

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
            switch (ptr->addr_space()) {
                case AddressSpace::Generic:  address_space = 0; break;
                case AddressSpace::Global:   address_space = 1; break;
                case AddressSpace::Texture:  address_space = 2; break;
                case AddressSpace::Shared:   address_space = 3; break;
                case AddressSpace::Constant: address_space = 4; break;
                default:
                    THORIN_UNREACHABLE;
            }
            llvm_type = llvm::PointerType::get(convert(ptr->referenced_type()), address_space);
            break;
        }
        case Node_IndefiniteArrayType:
            return types_[*type] = llvm::ArrayType::get(convert(type.as<ArrayType>()->elem_type()), 0);
        case Node_DefiniteArrayType: {
            auto array = type.as<DefiniteArrayType>();
            return types_[*type] = llvm::ArrayType::get(convert(array->elem_type()), array->dim());
        }
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type.as<FnType>();
            llvm::Type* ret = nullptr;
            std::vector<llvm::Type*> args;
            for (auto arg : fn->args()) {
                if (arg.isa<MemType>())
                    continue;
                if (auto fn = arg.isa<FnType>()) {
                    assert(!ret && "only one 'return' supported");
                    if (fn->empty())
                        ret = llvm::Type::getVoidTy(context_);
                    else if (fn->num_args() == 1)
                        ret = fn->arg(0).isa<MemType>() ? llvm::Type::getVoidTy(context_) : convert(fn->arg(0));
                    else if (fn->num_args() == 2) {
                        if (fn->arg(0).isa<MemType>())
                            ret = convert(fn->arg(1));
                        else if (fn->arg(1).isa<MemType>())
                            ret = convert(fn->arg(0));
                        else
                            goto multiple;
                    } else {
multiple:
                        std::vector<llvm::Type*> args;
                        for (auto arg : fn->args()) {
                            if (!arg.isa<MemType>())
                                args.push_back(convert(arg));
                        }
                        ret = llvm::StructType::get(context_, args);
                    }
                } else
                    args.push_back(convert(arg));
            }
            assert(ret);

            return types_[*type] = llvm::FunctionType::get(ret, args, false);
        }

        case Node_StructAbsType:
            return types_[*type] = llvm::StructType::create(context_);

        case Node_StructAppType: {
            auto struct_app = type.as<StructAppType>();
            auto llvm_struct = llvm::cast<llvm::StructType>(convert(struct_app->struct_abs_type()));
            assert(!types_.contains(*struct_app) && "type already converted");
            // important: memoize before recursing into element types to avoid endless recursion
            types_[*struct_app] = llvm_struct;
            Array<llvm::Type*> llvm_types(struct_app->num_elems());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(struct_app->elem(i));
            llvm_struct->setBody(llvm_ref(llvm_types));
            return llvm_struct;
        }

        case Node_TupleType: {
            auto tuple = type.as<TupleType>();
            Array<llvm::Type*> llvm_types(tuple->num_args());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(tuple->arg(i));
            return types_[*tuple] = llvm::StructType::get(context_, llvm_ref(llvm_types));
        }

        default:
            THORIN_UNREACHABLE;
    }

    if (type->length() == 1)
        return types_[*type] = llvm_type;
    return types_[*type] = llvm::VectorType::get(llvm_type, type->length());
}

llvm::GlobalVariable* CodeGen::emit_global_memory(llvm::Type* type, const std::string& name, unsigned addr_space) {
    return new llvm::GlobalVariable(*module_, type, false,
            llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(type), name,
            nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
}

void CodeGen::create_loop(llvm::Value* lower, llvm::Value* upper, llvm::Value* increment, llvm::Function* entry, std::function<void(llvm::Value*)> fun) {
    auto head = llvm::BasicBlock::Create(context_, "head", entry);
    auto body = llvm::BasicBlock::Create(context_, "body", entry);
    auto exit = llvm::BasicBlock::Create(context_, "exit", entry);
    // create loop phi and connect init value
    auto loop_counter = llvm::PHINode::Create(builder_.getInt32Ty(), 2U, "parallel_loop_phi", head);
    loop_counter->addIncoming(lower, builder_.GetInsertBlock());
    // connect head
    builder_.CreateBr(head);
    builder_.SetInsertPoint(head);
    auto cond = builder_.CreateICmpSLT(loop_counter, upper);
    builder_.CreateCondBr(cond, body, exit);
    builder_.SetInsertPoint(body);

    // add instructions to the loop body
    fun(loop_counter);

    // inc loop counter
    loop_counter->addIncoming(builder_.CreateAdd(loop_counter, increment), body);
    builder_.CreateBr(head);
    builder_.SetInsertPoint(exit);
}

//------------------------------------------------------------------------------

void emit_llvm(World& world, int opt) {
    World cuda(world.name());
    World nvvm(world.name());
    World spir(world.name());
    World opencl(world.name());

    // determine different parts of the world which need to be compiled differently
    Scope::for_each(world, [&] (const Scope& scope) {
        auto lambda = scope.entry();
        Lambda* imported = nullptr;
        if (lambda->is_passed_to_intrinsic(Intrinsic::CUDA))
            imported = import(cuda, lambda)->as_lambda();
        else if (lambda->is_passed_to_intrinsic(Intrinsic::NVVM))
            imported = import(nvvm, lambda)->as_lambda();
        else if (lambda->is_passed_to_intrinsic(Intrinsic::SPIR))
            imported = import(spir, lambda)->as_lambda();
        else if (lambda->is_passed_to_intrinsic(Intrinsic::OpenCL))
            imported = import(opencl, lambda)->as_lambda();
        else
            return;

        imported->name = lambda->unique_name();
        imported->make_external();
        lambda->name = lambda->unique_name();
        lambda->destroy_body();

        for (size_t i = 0, e = lambda->num_params(); i != e; ++i)
            imported->param(i)->name = lambda->param(i)->unique_name();
    });

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
