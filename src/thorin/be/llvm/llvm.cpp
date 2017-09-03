#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <stdexcept>

#include <llvm/ADT/Triple.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>

#ifdef RV_SUPPORT
#include <rv/rv.h>
#endif

#include "thorin/def.h"
#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/llvm/amdgpu.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/opencl.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/importer.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

namespace thorin {

CodeGen::CodeGen(World& world, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention, const Cont2Config& kernel_config)
    : world_(world)
    , context_()
    , module_(new llvm::Module(world.name(), context_))
    , irbuilder_(context_)
    , dibuilder_(*module_.get())
    , function_calling_convention_(function_calling_convention)
    , device_calling_convention_(device_calling_convention)
    , kernel_calling_convention_(kernel_calling_convention)
    , kernel_config_(kernel_config)
    , runtime_(new Runtime(context_, *module_.get(), irbuilder_))
{}

Continuation* CodeGen::emit_intrinsic(Continuation* continuation) {
    auto callee = continuation->callee()->as_continuation();
    switch (callee->intrinsic()) {
        case Intrinsic::Atomic:    return emit_atomic(continuation);
        case Intrinsic::CmpXchg:   return emit_cmpxchg(continuation);
        case Intrinsic::Reserve:   return emit_reserve(continuation);
        case Intrinsic::CUDA:      return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM,   ".cu",   continuation);
        case Intrinsic::NVVM:      return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM,   ".nvvm", continuation);
        case Intrinsic::OpenCL:    return runtime_->emit_host_code(*this, Runtime::OPENCL_PLATFORM, ".cl",   continuation);
        case Intrinsic::AMDGPU:    return runtime_->emit_host_code(*this, Runtime::AMDGPU_PLATFORM, ".gcn",  continuation);
        case Intrinsic::Parallel:  return emit_parallel(continuation);
        case Intrinsic::Spawn:     return emit_spawn(continuation);
        case Intrinsic::Sync:      return emit_sync(continuation);
#ifdef RV_SUPPORT
        case Intrinsic::Vectorize: return emit_vectorize_continuation(continuation);
#else
        case Intrinsic::Vectorize: throw std::runtime_error("rebuild with RV support");
#endif
        default: THORIN_UNREACHABLE;
    }
}

void CodeGen::emit_result_phi(const Param* param, llvm::Value* value) {
    thorin::find(phis_, param)->addIncoming(value, irbuilder_.GetInsertBlock());
}

Continuation* CodeGen::emit_atomic(Continuation* continuation) {
    assert(continuation->num_args() == 5 && "required arguments are missing");
    if (!is_type_i(continuation->arg(3)->type()))
        ELOG(continuation->arg(3), "atomic only supported for integer types");
    // atomic tag: Xchg Add Sub And Nand Or Xor Max Min
    u32 tag = continuation->arg(1)->as<PrimLit>()->qu32_value();
    auto ptr = lookup(continuation->arg(2));
    auto val = lookup(continuation->arg(3));
    assert(int(llvm::AtomicRMWInst::BinOp::Xchg) <= int(tag) && int(tag) <= int(llvm::AtomicRMWInst::BinOp::UMin) && "unsupported atomic");
    auto binop = (llvm::AtomicRMWInst::BinOp)tag;
    auto cont = continuation->arg(4)->as_continuation();
    auto call = irbuilder_.CreateAtomicRMW(binop, ptr, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SynchronizationScope::CrossThread);
    emit_result_phi(cont->param(1), call);
    return cont;
}

Continuation* CodeGen::emit_cmpxchg(Continuation* continuation) {
    assert(continuation->num_args() == 5 && "required arguments are missing");
    if (!is_type_i(continuation->arg(3)->type()))
        ELOG(continuation->arg(3), "cmpxchg only supported for integer types");
    auto ptr  = lookup(continuation->arg(1));
    auto cmp  = lookup(continuation->arg(2));
    auto val  = lookup(continuation->arg(3));
    auto cont = continuation->arg(4)->as_continuation();
    auto call = irbuilder_.CreateAtomicCmpXchg(ptr, cmp, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SynchronizationScope::CrossThread);
    auto loaded  = irbuilder_.CreateExtractValue(call, unsigned(0));
    auto success = irbuilder_.CreateExtractValue(call, unsigned(1));
    emit_result_phi(cont->param(1), loaded);
    emit_result_phi(cont->param(2), success);
    return cont;
}

Continuation* CodeGen::emit_reserve(const Continuation* continuation) {
    ELOG(&continuation->jump_debug(), "reserve_shared: only allowed in device code");
    THORIN_UNREACHABLE;
}

Continuation* CodeGen::emit_reserve_shared(const Continuation* continuation, bool init_undef) {
    assert(continuation->num_args() == 3 && "required arguments are missing");
    if (!continuation->arg(1)->isa<PrimLit>())
        ELOG(continuation->arg(1), "reserve_shared: couldn't extract memory size");
    auto num_elems = continuation->arg(1)->as<PrimLit>()->ps32_value();
    auto cont = continuation->arg(2)->as_continuation();
    auto type = convert(cont->param(1)->type());
    // construct array type
    auto elem_type = cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
    auto smem_type = this->convert(continuation->world().definite_array_type(elem_type, num_elems));
    auto name = continuation->unique_name();
    // NVVM doesn't allow '.' in global identifier
    std::replace(name.begin(), name.end(), '.', '_');
    auto global = emit_global_variable(smem_type, name, 3, init_undef);
    auto call = irbuilder_.CreatePointerCast(global, type);
    emit_result_phi(cont->param(1), call);
    return cont;
}

llvm::Value* CodeGen::emit_bitcast(const Def* val, const Type* dst_type) {
    auto from = lookup(val);
    auto src_type = val->type();
    auto to = convert(dst_type);
    if (src_type->isa<PtrType>() && dst_type->isa<PtrType>())
        return irbuilder_.CreatePointerCast(from, to);
    return irbuilder_.CreateBitCast(from, to);
}

llvm::FunctionType* CodeGen::convert_fn_type(Continuation* continuation) {
    return llvm::cast<llvm::FunctionType>(convert(continuation->type()));
}

llvm::Function* CodeGen::emit_function_decl(Continuation* continuation) {
    if (auto f = thorin::find(fcts_, continuation))
        return f;

    std::string name = (continuation->is_external() || continuation->empty()) ? continuation->name() : continuation->unique_name();
    auto f = llvm::cast<llvm::Function>(module_->getOrInsertFunction(name, convert_fn_type(continuation)));

#ifdef _MSC_VER
    // set dll storage class for MSVC
    if (!entry_ && llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
        if (continuation->empty()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        } else if (continuation->is_external()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }
#endif

    // set linkage
    if (continuation->empty() || continuation->is_external())
        f->setLinkage(llvm::Function::ExternalLinkage);
    else
        f->setLinkage(llvm::Function::InternalLinkage);

    // set calling convention
    if (continuation->is_external()) {
        f->setCallingConv(kernel_calling_convention_);
        emit_function_decl_hook(continuation, f);
    } else {
        if (continuation->cc() == CC::Device)
            f->setCallingConv(device_calling_convention_);
        else
            f->setCallingConv(function_calling_convention_);
    }

    return fcts_[continuation] = f;
}

void CodeGen::emit(int opt, bool debug) {
    llvm::DICompileUnit* dicompile_unit;
    if (debug) {
        module_->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
        // Darwin only supports dwarf2
        if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin())
            module_->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
        dicompile_unit = dibuilder_.createCompileUnit(llvm::dwarf::DW_LANG_C, dibuilder_.createFile(world_.name(), llvm::StringRef()), "Impala", opt > 0, llvm::StringRef(), 0);
    }

    Scope::for_each(world_, [&] (const Scope& scope) {
        entry_ = scope.entry();
        assert(entry_->is_returning());
        llvm::Function* fct = emit_function_decl(entry_);

        llvm::DISubprogram* disub_program;
        llvm::DIScope* discope = dicompile_unit;
        if (debug) {
            auto src_file = llvm::sys::path::filename(entry_->location().filename());
            auto src_dir = llvm::sys::path::parent_path(entry_->location().filename());
            auto difile = dibuilder_.createFile(src_file, src_dir);
            disub_program = dibuilder_.createFunction(discope, fct->getName(), fct->getName(), difile, entry_->location().front_line(),
                                                      dibuilder_.createSubroutineType(dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
                                                      false /* internal linkage */, true /* definition */, entry_->location().front_line(),
                                                      llvm::DINode::FlagPrototyped /* Flags */, opt > 0);
            fct->setSubprogram(disub_program);
            discope = disub_program;
        }

        // map params
        const Param* ret_param = nullptr;
        auto arg = fct->arg_begin();
        for (auto param : entry_->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                auto argv = &*arg;
                auto value = map_param(fct, argv, param);
                if (value == argv) {
                    arg->setName(param->unique_name()); // use param
                    params_[param] = &*arg++;
                } else {
                    params_[param] = value;             // use provided value
                }
            } else {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        BBMap bb2continuation;
        Schedule schedule(scope);

        for (const auto& block : schedule) {
            auto continuation = block.continuation();
            // map all bb-like continuations to llvm bb stubs
            if (continuation->intrinsic() != Intrinsic::EndScope) {
                auto bb = bb2continuation[continuation] = llvm::BasicBlock::Create(context_, continuation->name(), fct);

                // create phi node stubs (for all continuations different from entry)
                if (entry_ != continuation) {
                    for (auto param : continuation->params()) {
                        if (!is_mem(param) && !is_unit(param)) {
                            auto phi = llvm::PHINode::Create(convert(param->type()), (unsigned) param->peek().size(), param->name(), bb);
                            phis_[param] = phi;
                        }
                    }
                }
            }
        }

        auto oldStartBB = fct->begin();
        auto startBB = llvm::BasicBlock::Create(context_, fct->getName() + "_start", fct, &*oldStartBB);
        irbuilder_.SetInsertPoint(startBB);
        if (debug)
            irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(entry_->location().front_line(), entry_->location().front_col(), discope));
        emit_function_start(startBB, entry_);
        irbuilder_.CreateBr(&*oldStartBB);

        for (auto& block : schedule) {
            auto continuation = block.continuation();
            if (continuation->intrinsic() == Intrinsic::EndScope)
                continue;
            assert(continuation == entry_ || continuation->is_basicblock());
            irbuilder_.SetInsertPoint(bb2continuation[continuation]);

            for (auto primop : block) {
                if (debug)
                    irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(primop->location().front_line(), primop->location().front_col(), discope));

                if (primop->type()->order() >= 1) {
                    // ignore higher-order primops which come from a match intrinsic
                    if (is_from_match(primop)) continue;
                    THORIN_UNREACHABLE;
                }

                auto llvm_value = emit(primop);
                primops_[primop] = llvm_value;
            }

            // terminate bb
            if (debug)
                irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(continuation->jump_debug().front_line(), continuation->jump_debug().front_col(), discope));
            if (continuation->callee() == ret_param) { // return
                size_t num_args = continuation->num_args();
                if (num_args == 0) irbuilder_.CreateRetVoid();
                else {
                    Array<llvm::Value*> values(num_args);
                    Array<llvm::Type*> args(num_args);

                    size_t n = 0;
                    for (auto arg : continuation->args()) {
                        if (!is_mem(arg) && !is_unit(arg)) {
                            auto val = lookup(arg);
                            values[n] = val;
                            args[n++] = val->getType();
                        }
                    }

                    if (n == 0) irbuilder_.CreateRetVoid();
                    else if (n == 1) irbuilder_.CreateRet(values[0]);
                    else {
                        values.shrink(n);
                        args.shrink(n);
                        llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context_, llvm_ref(args)));

                        for (size_t i = 0; i != n; ++i)
                            agg = irbuilder_.CreateInsertValue(agg, values[i], { unsigned(i) });

                        irbuilder_.CreateRet(agg);
                    }
                }
            } else if (continuation->callee() == world().branch()) {
                auto cond = lookup(continuation->arg(0));
                auto tbb = bb2continuation[continuation->arg(1)->as_continuation()];
                auto fbb = bb2continuation[continuation->arg(2)->as_continuation()];
                irbuilder_.CreateCondBr(cond, tbb, fbb);
            } else if (continuation->callee()->isa<Continuation>() &&
                       continuation->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
                auto val = lookup(continuation->arg(0));
                auto otherwise_bb = bb2continuation[continuation->arg(1)->as_continuation()];
                auto match = irbuilder_.CreateSwitch(val, otherwise_bb, continuation->num_args() - 2);
                for (size_t i = 2; i < continuation->num_args(); i++) {
                    auto arg = continuation->arg(i)->as<Tuple>();
                    auto case_const = llvm::cast<llvm::ConstantInt>(lookup(arg->op(0)));
                    auto case_bb    = bb2continuation[arg->op(1)->as_continuation()];
                    match->addCase(case_const, case_bb);
                }
            } else if (continuation->callee()->isa<Bottom>()) {
                irbuilder_.CreateUnreachable();
            } else {
                auto callee = continuation->callee()->as_continuation();
                if (callee->is_basicblock())         // ordinary jump
                    irbuilder_.CreateBr(bb2continuation[callee]);
                else {
                    if (callee->is_intrinsic()) {
                        auto ret_continuation = emit_intrinsic(continuation);
                        irbuilder_.CreateBr(bb2continuation[ret_continuation]);
                    } else {
                        // put all first-order args into an array
                        std::vector<llvm::Value*> args;
                        const Def* ret_arg = nullptr;
                        for (auto arg : continuation->args()) {
                            if (arg->order() == 0) {
                                if (!is_mem(arg) && !is_unit(arg))
                                    args.push_back(lookup(arg));
                            } else {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }

                        llvm::CallInst* call = irbuilder_.CreateCall(emit_function_decl(callee), args);
                        if (callee->is_external())
                            call->setCallingConv(kernel_calling_convention_);
                        else if (callee->cc() == CC::Device)
                            call->setCallingConv(device_calling_convention_);
                        else
                            call->setCallingConv(function_calling_convention_);

                        // must be call + continuation --- call + return has been removed by codegen_prepare
                        auto succ = ret_arg->as_continuation();

                        size_t n = 0;
                        const Param* last_param = nullptr;
                        for (auto param : succ->params()) {
                            if (is_mem(param) || is_unit(param))
                                continue;
                            last_param = param;
                            n++;
                        }

                        if (n == 0) {
                            irbuilder_.CreateBr(bb2continuation[succ]);
                        } else if (n == 1) {
                            irbuilder_.CreateBr(bb2continuation[succ]);
                            emit_result_phi(last_param, call);
                        } else {
                            Array<llvm::Value*> extracts(n);
                            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                                auto param = succ->param(i);
                                if (is_mem(param) || is_unit(param))
                                    continue;
                                extracts[j] = irbuilder_.CreateExtractValue(call, unsigned(j));
                                j++;
                            }

                            irbuilder_.CreateBr(bb2continuation[succ]);

                            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                                auto param = succ->param(i);
                                if (is_mem(param) || is_unit(param))
                                    continue;
                                emit_result_phi(param, extracts[j]);
                                j++;
                            }
                        }
                    }
                }
            }
        }

        // add missing arguments to phis_
        for (const auto& p : phis_) {
            auto param = p.first;
            auto phi = p.second;

            for (const auto& peek : param->peek())
                phi->addIncoming(lookup(peek.def()), bb2continuation[peek.from()]);
        }

        params_.clear();
        phis_.clear();
        primops_.clear();
    });

#ifdef RV_SUPPORT
    // emit vectorized code
    for (const auto& tuple : vec_todo_)
        emit_vectorize(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple));
    vec_todo_.clear();

    rv::lowerIntrinsics(*module_);
#endif

#ifndef NDEBUG
    llvm::verifyModule(*module_);
#endif
    optimize(opt);
    if (debug)
        dibuilder_.finalize();

    std::error_code EC;
    auto ll_name = get_output_name(world_.name());
    llvm::raw_fd_ostream out(ll_name, EC, llvm::sys::fs::F_Text);
    if (EC)
        throw std::runtime_error("cannot write '" + ll_name + "': " + EC.message());

    module_->print(out, nullptr);
}

void CodeGen::optimize(int opt) {
    if (opt != 0) {
        llvm::PassManagerBuilder pmbuilder;
        llvm::legacy::PassManager pass_manager;
        if (opt == -1) {
            pmbuilder.OptLevel = 2u;
            pmbuilder.SizeLevel = 1;
        } else {
            pmbuilder.OptLevel = (unsigned) opt;
            pmbuilder.SizeLevel = 0u;
        }
        if (opt == 3) {
            pass_manager.add(llvm::createFunctionInliningPass());
            pass_manager.add(llvm::createAggressiveDCEPass());
        }
        pmbuilder.populateModulePassManager(pass_manager);

        pass_manager.run(*module_);
    }
}

llvm::Value* CodeGen::lookup(const Def* def) {
    if (auto primop = def->isa<PrimOp>()) {
        if (auto res = thorin::find(primops_, primop))
            return res;
        else {
            auto llvm_value = emit(def);
            return primops_[primop] = llvm_value;
        }
    }

    if (auto param = def->isa<Param>()) {
        auto i = params_.find(param);
        if (i != params_.end())
            return i->second;

        assert(phis_.find(param) != phis_.end());
        return thorin::find(phis_, param);
    }

    THORIN_UNREACHABLE;
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::Type* type, const std::string& name) {
    auto entry = &irbuilder_.GetInsertBlock()->getParent()->getEntryBlock();
    llvm::AllocaInst* alloca;
    if (entry->empty())
        alloca = new llvm::AllocaInst(type, nullptr, name, entry);
    else
        alloca = new llvm::AllocaInst(type, nullptr, name, entry->getFirstNonPHIOrDbg());
    return alloca;
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (auto bin = def->isa<BinOp>()) {
        llvm::Value* lhs = lookup(bin->lhs());
        llvm::Value* rhs = lookup(bin->rhs());
        const std::string& name = bin->name();

        if (auto cmp = bin->isa<Cmp>()) {
            auto type = cmp->lhs()->type();
            if (is_type_s(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateICmpSGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateICmpSGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateICmpSLT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateICmpSLE(lhs, rhs, name);
                }
            } else if (is_type_u(type) || is_type_bool(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateICmpUGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateICmpUGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateICmpULT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateICmpULE(lhs, rhs, name);
                }
            } else if (is_type_f(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder_.CreateFCmpOEQ(lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateFCmpUNE(lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateFCmpOGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateFCmpOGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateFCmpOLT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateFCmpOLE(lhs, rhs, name);
                }
            } else if (type->isa<PtrType>()) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    default: THORIN_UNREACHABLE;
                }
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();
            bool q = is_type_q(arithop->type()); // quick? -> nsw/nuw/fast float

            if (is_type_f(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder_.CreateFAdd(lhs, rhs, name);
                    case ArithOp_sub: return irbuilder_.CreateFSub(lhs, rhs, name);
                    case ArithOp_mul: return irbuilder_.CreateFMul(lhs, rhs, name);
                    case ArithOp_div: return irbuilder_.CreateFDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder_.CreateFRem(lhs, rhs, name);
                    case ArithOp_and:
                    case ArithOp_or:
                    case ArithOp_xor:
                    case ArithOp_shl:
                    case ArithOp_shr: THORIN_UNREACHABLE;
                }
            }

            if (is_type_s(type) || is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder_.CreateAdd (lhs, rhs, name, false, q);
                    case ArithOp_sub: return irbuilder_.CreateSub (lhs, rhs, name, false, q);
                    case ArithOp_mul: return irbuilder_.CreateMul (lhs, rhs, name, false, q);
                    case ArithOp_div: return irbuilder_.CreateSDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder_.CreateSRem(lhs, rhs, name);
                    case ArithOp_and: return irbuilder_.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return irbuilder_.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return irbuilder_.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return irbuilder_.CreateShl (lhs, rhs, name, false, q);
                    case ArithOp_shr: return irbuilder_.CreateAShr(lhs, rhs, name);
                }
            }
            if (is_type_u(type) || is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder_.CreateAdd (lhs, rhs, name, q, false);
                    case ArithOp_sub: return irbuilder_.CreateSub (lhs, rhs, name, q, false);
                    case ArithOp_mul: return irbuilder_.CreateMul (lhs, rhs, name, q, false);
                    case ArithOp_div: return irbuilder_.CreateUDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder_.CreateURem(lhs, rhs, name);
                    case ArithOp_and: return irbuilder_.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return irbuilder_.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return irbuilder_.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return irbuilder_.CreateShl (lhs, rhs, name, q, false);
                    case ArithOp_shr: return irbuilder_.CreateLShr(lhs, rhs, name);
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
            if (src_type->isa<PtrType>() && dst_type->isa<PtrType>()) {
                return irbuilder_.CreatePointerCast(from, to);
            }
            if (src_type->isa<PtrType>()) {
                assert(is_type_i(dst_type) || is_type_bool(dst_type));
                return irbuilder_.CreatePtrToInt(from, to);
            }
            if (dst_type->isa<PtrType>()) {
                assert(is_type_i(src_type) || is_type_bool(src_type));
                return irbuilder_.CreateIntToPtr(from, to);
            }

            auto src = src_type->as<PrimType>();
            auto dst = dst_type->as<PrimType>();

            if (is_type_f(src) && is_type_f(dst)) {
                assert(num_bits(src->primtype_tag()) != num_bits(dst->primtype_tag()));
                return irbuilder_.CreateFPCast(from, to);
            }
            if (is_type_f(src)) {
                if (is_type_s(dst))
                    return irbuilder_.CreateFPToSI(from, to);
                return irbuilder_.CreateFPToUI(from, to);
            }
            if (is_type_f(dst)) {
                if (is_type_s(src))
                    return irbuilder_.CreateSIToFP(from, to);
                return irbuilder_.CreateUIToFP(from, to);
            }

            if (num_bits(src->primtype_tag()) > num_bits(dst->primtype_tag())) {
                if (is_type_i(src) && (is_type_i(dst) || is_type_bool(dst)))
                    return irbuilder_.CreateTrunc(from, to);
            } else if (num_bits(src->primtype_tag()) < num_bits(dst->primtype_tag())) {
                if ( is_type_s(src)                       && is_type_i(dst)) return irbuilder_.CreateSExt(from, to);
                if ((is_type_u(src) || is_type_bool(src)) && is_type_i(dst)) return irbuilder_.CreateZExt(from, to);
            }

            assert(false && "unsupported cast");
        }

        if (conv->isa<Bitcast>())
            return emit_bitcast(conv->from(), dst_type);
    }

    if (auto select = def->isa<Select>()) {
        if (def->type()->isa<FnType>())
            return nullptr;

        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return irbuilder_.CreateSelect(cond, tval, fval);
    }

    if (auto size_of = def->isa<SizeOf>()) {
        auto type = convert(size_of->of());
        auto layout = llvm::DataLayout(module_->getDataLayout());
        return irbuilder_.getInt32(layout.getTypeAllocSize(type));
    }

    if (auto array = def->isa<DefiniteArray>()) {
        auto type = llvm::cast<llvm::ArrayType>(convert(array->type()));
        if (is_const(array)) {
            size_t size = array->num_ops();
            Array<llvm::Constant*> vals(size);
            for (size_t i = 0; i != size; ++i)
                vals[i] = llvm::cast<llvm::Constant>(emit(array->op(i)));
            return llvm::ConstantArray::get(type, llvm_ref(vals));
        }
        WLOG(def, "slow: alloca and loads/stores needed for definite array '{}'", def);
        auto alloca = emit_alloca(type, array->name());

        u64 i = 0;
        llvm::Value* args[2] = { irbuilder_.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = irbuilder_.getInt64(i++);
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args, op->name());
            irbuilder_.CreateStore(lookup(op), gep);
        }

        return irbuilder_.CreateLoad(alloca);
    }

    if (auto array = def->isa<IndefiniteArray>())
        return llvm::UndefValue::get(convert(array->type()));

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>());
        llvm::Value* llvm_agg = llvm::UndefValue::get(convert(agg->type()));

        if (def->isa<Vector>()) {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder_.CreateInsertElement(llvm_agg, lookup(agg->op(i)), irbuilder_.getInt32(i));
        } else {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder_.CreateInsertValue(llvm_agg, lookup(agg->op(i)), { unsigned(i) });
        }

        return llvm_agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto llvm_agg = lookup(aggop->agg());
        auto llvm_idx = lookup(aggop->index());
        auto copy_to_alloca = [&] () {
            WLOG(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto alloca = emit_alloca(llvm_agg->getType(), aggop->name());
            irbuilder_.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { irbuilder_.getInt64(0), llvm_idx };
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args);
            return std::make_pair(alloca, gep);
        };

        if (auto extract = aggop->isa<Extract>()) {
            // Assemblys with more than two outputs are MemOps and have tuple type
            // and thus need their own rule here because the standard MemOp rule does not work
            if (auto assembly = extract->agg()->isa<Assembly>()) {
                if (assembly->type()->num_ops() > 2 && primlit_value<unsigned>(aggop->index()) != 0)
                    return irbuilder_.CreateExtractValue(llvm_agg, {primlit_value<unsigned>(aggop->index()) - 1});
            }

            if (auto memop = extract->agg()->isa<MemOp>())
                return lookup(memop);

            if (aggop->agg()->type()->isa<ArrayType>())
                return irbuilder_.CreateLoad(copy_to_alloca().second);

            if (extract->agg()->type()->isa<VectorType>())
                return irbuilder_.CreateExtractElement(llvm_agg, llvm_idx);
            // tuple/struct
            return irbuilder_.CreateExtractValue(llvm_agg, {primlit_value<unsigned>(aggop->index())});
        }

        auto insert = def->as<Insert>();
        auto value = lookup(insert->value());

        if (insert->agg()->type()->isa<ArrayType>()) {
            auto p = copy_to_alloca();
            irbuilder_.CreateStore(lookup(aggop->as<Insert>()->value()), p.second);
            return irbuilder_.CreateLoad(p.first);
        }
        if (insert->agg()->type()->isa<VectorType>())
            return irbuilder_.CreateInsertElement(llvm_agg, lookup(aggop->as<Insert>()->value()), llvm_idx);
        // tuple/struct
        return irbuilder_.CreateInsertValue(llvm_agg, value, {primlit_value<unsigned>(aggop->index())});
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* llvm_type = convert(primlit->type());
        Box box = primlit->value();

        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     return irbuilder_. getInt1(box.get_bool());
            case PrimType_ps8:  case PrimType_qs8:  return irbuilder_. getInt8(box. get_s8());
            case PrimType_pu8:  case PrimType_qu8:  return irbuilder_. getInt8(box. get_u8());
            case PrimType_ps16: case PrimType_qs16: return irbuilder_.getInt16(box.get_s16());
            case PrimType_pu16: case PrimType_qu16: return irbuilder_.getInt16(box.get_u16());
            case PrimType_ps32: case PrimType_qs32: return irbuilder_.getInt32(box.get_s32());
            case PrimType_pu32: case PrimType_qu32: return irbuilder_.getInt32(box.get_u32());
            case PrimType_ps64: case PrimType_qs64: return irbuilder_.getInt64(box.get_s64());
            case PrimType_pu64: case PrimType_qu64: return irbuilder_.getInt64(box.get_u64());
            case PrimType_pf16: case PrimType_qf16: return llvm::ConstantFP::get(llvm_type, box.get_f16());
            case PrimType_pf32: case PrimType_qf32: return llvm::ConstantFP::get(llvm_type, box.get_f32());
            case PrimType_pf64: case PrimType_qf64: return llvm::ConstantFP::get(llvm_type, box.get_f64());
        }
    }

    if (auto bottom = def->isa<Bottom>())
        return llvm::UndefValue::get(convert(bottom->type()));

    if (auto alloc = def->isa<Alloc>()) { // TODO factor this code
        auto llvm_malloc = runtime_->get(get_alloc_name().c_str());
        auto alloced_type = convert(alloc->alloced_type());
        llvm::CallInst* void_ptr;
        auto layout = module_->getDataLayout();
        if (auto array = alloc->alloced_type()->isa<IndefiniteArrayType>()) {
            auto size = irbuilder_.CreateAdd(
                    irbuilder_.getInt64(layout.getTypeAllocSize(alloced_type)),
                    irbuilder_.CreateMul(irbuilder_.CreateIntCast(lookup(alloc->extra()), irbuilder_.getInt64Ty(), false),
                                         irbuilder_.getInt64(layout.getTypeAllocSize(convert(array->elem_type())))));
            llvm::Value* malloc_args[] = { irbuilder_.getInt32(0), size };
            void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
        } else {
            llvm::Value* malloc_args[] = { irbuilder_.getInt32(0), irbuilder_.getInt64(layout.getTypeAllocSize(alloced_type)) };
            void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
        }

        return irbuilder_.CreatePointerCast(void_ptr, convert(alloc->out_ptr_type()));
    }

    if (auto load = def->isa<Load>())           return emit_load(load);
    if (auto store = def->isa<Store>())         return emit_store(store);
    if (auto lea = def->isa<LEA>())             return emit_lea(lea);
    if (auto assembly = def->isa<Assembly>())   return emit_assembly(assembly);
    if (def->isa<Enter>())                      return nullptr;

    if (auto slot = def->isa<Slot>())
        return irbuilder_.CreateAlloca(convert(slot->type()->as<PtrType>()->pointee()), 0, slot->unique_name());

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(convert(vector->type()));
        for (size_t i = 0, e = vector->num_ops(); i != e; ++i)
            vec = irbuilder_.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world_.literal_pu32(i, vector->location())));

        return vec;
    }

    if (auto global = def->isa<Global>())
        return emit_global(global);

    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_global(const Global* global) {
    llvm::Value* val;
    if (auto continuation = global->init()->isa_continuation())
        val = fcts_[continuation];
    else {
        auto llvm_type = convert(global->alloced_type());
        auto var = llvm::cast<llvm::GlobalVariable>(module_->getOrInsertGlobal(global->name(), llvm_type));
        if (global->init()->isa<Bottom>())
            var->setInitializer(llvm::Constant::getNullValue(llvm_type)); // HACK
        else
            var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));
        val = var;
    }
    return val;
}

llvm::Value* CodeGen::emit_load(const Load* load) {
    return irbuilder_.CreateLoad(lookup(load->ptr()));
}

llvm::Value* CodeGen::emit_store(const Store* store) {
    return irbuilder_.CreateStore(lookup(store->val()), lookup(store->ptr()));
}

llvm::Value* CodeGen::emit_lea(const LEA* lea) {
    if (lea->ptr_pointee()->isa<TupleType>() || lea->ptr_pointee()->isa<StructType>())
        return irbuilder_.CreateStructGEP(convert(lea->ptr_pointee()), lookup(lea->ptr()), primlit_value<u32>(lea->index()));

    assert(lea->ptr_pointee()->isa<ArrayType>());
    llvm::Value* args[2] = { irbuilder_.getInt64(0), lookup(lea->index()) };
    return irbuilder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
}

llvm::Value* CodeGen::emit_assembly(const Assembly* assembly) {
    const TupleType *out_type = assembly->type();
    llvm::Type *res_type;
    switch (out_type->num_ops()) {
        case 0:
            THORIN_UNREACHABLE;
            // there must always be the mem type as output
        case 1:
            res_type = llvm::Type::getVoidTy(context_);
            break;
        case 2:
            res_type = convert(assembly->type()->op(1));
            break;
        default:
            res_type = convert(world().tuple_type(assembly->type()->ops().skip_front()));
            break;
    }

    size_t num_inputs = assembly->num_inputs();
    auto input_values = Array<llvm::Value*>(num_inputs);
    auto input_types = Array<llvm::Type*>(num_inputs);
    for (size_t i = 0; i != num_inputs; ++i) {
        input_values[i] = lookup(assembly->input(i));
        input_types[i] = convert(assembly->input(i)->type());
    }

    auto *fn_type = llvm::FunctionType::get(res_type, llvm_ref(input_types), false);

    std::string constraints;
    for (auto con : assembly->output_constraints())
        constraints += con + ",";
    for (auto con : assembly->input_constraints())
        constraints += con + ",";
    for (auto clob : assembly->clobbers())
        constraints += "~{" + clob + "},";
    // clang always marks those registers as clobbered, so we will do so as well
    constraints += "~{dirflag},~{fpsr},~{flags}";

    if (!llvm::InlineAsm::Verify(fn_type, constraints))
        ELOG(assembly, "constraints and input and output types of inline assembly do not match");

    auto asm_expr = llvm::InlineAsm::get(fn_type, assembly->asm_template(), constraints,
            assembly->has_sideeffects(), assembly->is_alignstack(),
            assembly->is_inteldialect() ? llvm::InlineAsm::AsmDialect::AD_Intel : llvm::InlineAsm::AsmDialect::AD_ATT);
    return irbuilder_.CreateCall(asm_expr, llvm_ref(input_values));
}

unsigned CodeGen::convert_addr_space(const AddrSpace addr_space) {
    switch (addr_space) {
        case AddrSpace::Generic:  return 0;
        case AddrSpace::Global:   return 1;
        case AddrSpace::Texture:  return 2;
        case AddrSpace::Shared:   return 3;
        case AddrSpace::Constant: return 4;
        default:                  THORIN_UNREACHABLE;
    }
}

llvm::Type* CodeGen::convert(const Type* type) {
    if (auto llvm_type = thorin::find(types_, type))
        return llvm_type;

    assert(!type->isa<MemType>());
    llvm::Type* llvm_type;
    switch (type->tag()) {
        case PrimType_bool:                                                             llvm_type = irbuilder_. getInt1Ty();  break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  llvm_type = irbuilder_. getInt8Ty();  break;
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: llvm_type = irbuilder_.getInt16Ty();  break;
        case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: llvm_type = irbuilder_.getInt32Ty();  break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: llvm_type = irbuilder_.getInt64Ty();  break;
        case PrimType_pf16: case PrimType_qf16:                                         llvm_type = irbuilder_.getHalfTy();   break;
        case PrimType_pf32: case PrimType_qf32:                                         llvm_type = irbuilder_.getFloatTy();  break;
        case PrimType_pf64: case PrimType_qf64:                                         llvm_type = irbuilder_.getDoubleTy(); break;
        case Node_PtrType: {
            auto ptr = type->as<PtrType>();
            llvm_type = llvm::PointerType::get(convert(ptr->pointee()), convert_addr_space(ptr->addr_space()));
            break;
        }
        case Node_IndefiniteArrayType: {
            llvm_type = llvm::ArrayType::get(convert(type->as<ArrayType>()->elem_type()), 0);
            return types_[type] = llvm_type;
        }
        case Node_DefiniteArrayType: {
            auto array = type->as<DefiniteArrayType>();
            llvm_type = llvm::ArrayType::get(convert(array->elem_type()), array->dim());
            return types_[type] = llvm_type;
        }
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type->as<FnType>();
            llvm::Type* ret = nullptr;
            std::vector<llvm::Type*> ops;
            for (auto op : fn->ops()) {
                if (op->isa<MemType>() || op == world().unit()) continue;
                if (auto fn = op->isa<FnType>()) {
                    assert(!ret && "only one 'return' supported");
                    std::vector<llvm::Type*> ret_types;
                    for (auto fn_op : fn->ops()) {
                        if (fn_op->isa<MemType>() || fn_op == world().unit()) continue;
                        ret_types.push_back(convert(fn_op));
                    }
                    if (ret_types.size() == 0)      ret = llvm::Type::getVoidTy(context_);
                    else if (ret_types.size() == 1) ret = ret_types.back();
                    else                            ret = llvm::StructType::get(context_, ret_types);
                } else
                    ops.push_back(convert(op));
            }
            assert(ret);

            llvm_type = llvm::FunctionType::get(ret, ops, false);
            return types_[type] = llvm_type;
        }

        case Node_StructType: {
            auto struct_type = type->as<StructType>();
            auto llvm_struct = llvm::StructType::create(context_);

            // important: memoize before recursing into element types to avoid endless recursion
            assert(!types_.contains(struct_type) && "type already converted");
            types_[struct_type] = llvm_struct;

            Array<llvm::Type*> llvm_types(struct_type->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(struct_type->op(i));
            llvm_struct->setBody(llvm_ref(llvm_types));
            return llvm_struct;
        }

        case Node_TupleType: {
            auto tuple = type->as<TupleType>();
            Array<llvm::Type*> llvm_types(tuple->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(tuple->op(i));
            llvm_type = llvm::StructType::get(context_, llvm_ref(llvm_types));
            return types_[tuple] = llvm_type;
        }

        default:
            THORIN_UNREACHABLE;
    }

    if (vector_length(type) == 1)
        return types_[type] = llvm_type;

    llvm_type = llvm::VectorType::get(llvm_type, vector_length(type));
    return types_[type] = llvm_type;
}

llvm::GlobalVariable* CodeGen::emit_global_variable(llvm::Type* type, const std::string& name, unsigned addr_space, bool init_undef) {
    auto init = init_undef ? llvm::UndefValue::get(type) : llvm::Constant::getNullValue(type);
    return new llvm::GlobalVariable(*module_, type, false, llvm::GlobalValue::InternalLinkage, init, name, nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
}

void CodeGen::create_loop(llvm::Value* lower, llvm::Value* upper, llvm::Value* increment, llvm::Function* entry, std::function<void(llvm::Value*)> fun) {
    auto head = llvm::BasicBlock::Create(context_, "head", entry);
    auto body = llvm::BasicBlock::Create(context_, "body", entry);
    auto exit = llvm::BasicBlock::Create(context_, "exit", entry);
    // create loop phi and connect init value
    auto loop_counter = llvm::PHINode::Create(irbuilder_.getInt32Ty(), 2U, "parallel_loop_phi", head);
    loop_counter->addIncoming(lower, irbuilder_.GetInsertBlock());
    // connect head
    irbuilder_.CreateBr(head);
    irbuilder_.SetInsertPoint(head);
    auto cond = irbuilder_.CreateICmpSLT(loop_counter, upper);
    irbuilder_.CreateCondBr(cond, body, exit);
    irbuilder_.SetInsertPoint(body);

    // add instructions to the loop body
    fun(loop_counter);

    // inc loop counter
    loop_counter->addIncoming(irbuilder_.CreateAdd(loop_counter, increment), body);
    irbuilder_.CreateBr(head);
    irbuilder_.SetInsertPoint(exit);
}

//------------------------------------------------------------------------------

void emit_llvm(World& world, int opt, bool debug) {
    Importer cuda(world.name());
    Importer nvvm(world.name());
    Importer opencl(world.name());
    Importer amdgpu(world.name());

    Cont2Config kernel_config;

    // determine different parts of the world which need to be compiled differently
    Scope::for_each(world, [&] (const Scope& scope) {
        auto continuation = scope.entry();
        Continuation* imported = nullptr;
        if (is_passed_to_intrinsic(continuation, Intrinsic::CUDA))
            imported = cuda.import(continuation)->as_continuation();
        else if (is_passed_to_intrinsic(continuation, Intrinsic::NVVM))
            imported = nvvm.import(continuation)->as_continuation();
        else if (is_passed_to_intrinsic(continuation, Intrinsic::OpenCL))
            imported = opencl.import(continuation)->as_continuation();
        else if (is_passed_to_intrinsic(continuation, Intrinsic::AMDGPU))
            imported = amdgpu.import(continuation)->as_continuation();
        else
            return;

        imported->debug().set(continuation->unique_name());
        imported->make_external();
        continuation->debug().set(continuation->unique_name());
        continuation->destroy_body();

        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->debug().set(continuation->param(i)->unique_name());

        visit_uses(continuation, [&] (Continuation* use) {
            auto it_config = use->arg(LaunchArgs::Config)->as<Tuple>();
            if (it_config->op(0)->isa<PrimLit>() && it_config->op(1)->isa<PrimLit>() && it_config->op(2)->isa<PrimLit>()) {
                auto p = kernel_config.emplace(imported, std::tuple<int, int, int>{ it_config->op(0)->as<PrimLit>()->qu32_value().data(), it_config->op(1)->as<PrimLit>()->qu32_value().data(), it_config->op(2)->as<PrimLit>()->qu32_value().data() } );
                assert_unused(p.second && "expected only single entry");
            }
            return false;
        });

    });

    if (!cuda.world().empty() || !nvvm.world().empty() || !amdgpu.world().empty() || !opencl.world().empty()) {
        world.cleanup();
        codegen_prepare(world);
    }

    CPUCodeGen(world, kernel_config).emit(opt, debug);
    if (!cuda.  world().empty()) CUDACodeGen  (cuda  .world(), kernel_config).emit(/*opt,*/ debug);
    if (!nvvm.  world().empty()) NVVMCodeGen  (nvvm  .world(), kernel_config).emit(opt, debug);
    if (!opencl.world().empty()) OpenCLCodeGen(opencl.world(), kernel_config).emit(/*opt,*/ debug);
    if (!amdgpu.world().empty()) AMDGPUCodeGen(amdgpu.world(), kernel_config).emit(opt, debug);
}

//------------------------------------------------------------------------------

}
