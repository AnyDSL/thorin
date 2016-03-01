#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <stdexcept>

#include <llvm/ADT/Triple.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/PassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>

#ifdef WFV2_SUPPORT
#include <wfvInterface.h>
#endif

#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/opencl.h"
#include "thorin/be/llvm/spir.h"
#include "thorin/transform/import.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"
#include "thorin/util/push.h"

namespace thorin {

CodeGen::CodeGen(World& world, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention)
    : world_(world)
    , context_()
    , module_(new llvm::Module(world.name(), context_))
    , irbuilder_(context_)
    , dibuilder_(*module_.get())
    , function_calling_convention_(function_calling_convention)
    , device_calling_convention_(device_calling_convention)
    , kernel_calling_convention_(kernel_calling_convention)
    , runtime_(new Runtime(context_, *module_.get(), irbuilder_))
{}

Lambda* CodeGen::emit_intrinsic(Lambda* lambda) {
    Lambda* to = lambda->to()->as_lambda();
    switch (to->intrinsic()) {
        case Intrinsic::Atomic:     return emit_atomic(lambda);
        case Intrinsic::Select:     return emit_select(lambda);
        case Intrinsic::Sizeof:     return emit_sizeof(lambda);
        case Intrinsic::Shuffle:    return emit_shuffle(lambda);
        case Intrinsic::Reserve:    return emit_reserve(lambda);
        case Intrinsic::Bitcast:    return emit_reinterpret(lambda);
        case Intrinsic::CUDA:       return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM, ".cu", lambda);
        case Intrinsic::NVVM:       return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM, ".nvvm", lambda);
        case Intrinsic::SPIR:       return runtime_->emit_host_code(*this, Runtime::OPENCL_PLATFORM, ".spir.bc", lambda);
        case Intrinsic::OpenCL:     return runtime_->emit_host_code(*this, Runtime::OPENCL_PLATFORM, ".cl", lambda);
        case Intrinsic::Parallel:   return emit_parallel(lambda);
        case Intrinsic::Spawn:      return emit_spawn(lambda);
        case Intrinsic::Sync:       return emit_sync(lambda);
#ifdef WFV2_SUPPORT
        case Intrinsic::Vectorize:  return emit_vectorize_continuation(lambda);
#else
        case Intrinsic::Vectorize:  throw std::runtime_error("rebuild with libWFV support");
#endif
        default: THORIN_UNREACHABLE;
    }
}

void CodeGen::emit_result_phi(const Param* param, llvm::Value* value) {
    find(phis_, param)->addIncoming(value, irbuilder_.GetInsertBlock());
}

Lambda* CodeGen::emit_atomic(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    // atomic kind: Xchg Add Sub And Nand Or Xor Max Min
    u32 kind = lambda->arg(1)->as<PrimLit>()->qu32_value();
    auto ptr = lookup(lambda->arg(2));
    auto val = lookup(lambda->arg(3));
    assert(int(llvm::AtomicRMWInst::BinOp::Xchg) <= int(kind) && int(kind) <= int(llvm::AtomicRMWInst::BinOp::UMin) && "unsupported atomic");
    llvm::AtomicRMWInst::BinOp binop = (llvm::AtomicRMWInst::BinOp)kind;

    auto cont = lambda->arg(4)->as_lambda();
    auto call = irbuilder_.CreateAtomicRMW(binop, ptr, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SynchronizationScope::CrossThread);
    emit_result_phi(cont->param(1), call);
    return cont;
}

Lambda* CodeGen::emit_select(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    auto cond = lookup(lambda->arg(1));
    auto a = lookup(lambda->arg(2));
    auto b = lookup(lambda->arg(3));

    auto cont = lambda->arg(4)->as_lambda();
    auto call = irbuilder_.CreateSelect(cond, a, b);
    emit_result_phi(cont->param(1), call);
    return cont;
}

Lambda* CodeGen::emit_sizeof(Lambda* lambda) {
    assert(lambda->num_args() == 2 && "required arguments are missing");
    auto type = convert(lambda->type_arg(0));
    auto cont = lambda->arg(1)->as_lambda();
    auto layout = module_->getDataLayout();
    auto call = irbuilder_.getInt32(layout->getTypeAllocSize(type));
    emit_result_phi(cont->param(1), call);
    return cont;
}

Lambda* CodeGen::emit_shuffle(Lambda* lambda) {
    assert(lambda->num_args() == 5 && "required arguments are missing");
    auto mask = lookup(lambda->arg(3));
    auto a = lookup(lambda->arg(1));
    auto b = lookup(lambda->arg(2));

    auto cont = lambda->arg(4)->as_lambda();
    auto call = irbuilder_.CreateShuffleVector(a, b, mask);
    emit_result_phi(cont->param(1), call);
    return cont;
}

Lambda* CodeGen::emit_reserve(const Lambda* lambda) {
    ELOG("reserve_shared: only allowed in device code at %", lambda->jump_loc());
    THORIN_UNREACHABLE;
}

Lambda* CodeGen::emit_reserve_shared(const Lambda* lambda, bool prefix) {
    assert(lambda->num_args() == 3 && "required arguments are missing");
    if (!lambda->arg(1)->isa<PrimLit>())
        ELOG("reserve_shared: couldn't extract memory size at %", lambda->arg(1)->loc());
    auto num_elems = lambda->arg(1)->as<PrimLit>()->ps32_value();
    auto cont = lambda->arg(2)->as_lambda();
    auto type = convert(cont->param(1)->type());
    // construct array type
    auto elem_type = cont->param(1)->type().as<PtrType>()->referenced_type().as<ArrayType>()->elem_type();
    auto smem_type = this->convert(lambda->world().definite_array_type(elem_type, num_elems));
    auto global = emit_global_variable(smem_type, (prefix ? entry_->name + "." : "") + lambda->unique_name(), 3);
    auto call = irbuilder_.CreatePointerCast(global, type);
    emit_result_phi(cont->param(1), call);
    return cont;
}

llvm::Value* CodeGen::emit_bitcast(Def val, Type dst_type) {
    auto from = lookup(val);
    auto src_type = val->type();
    auto to = convert(dst_type);
    if (src_type.isa<PtrType>() && dst_type.isa<PtrType>())
        return irbuilder_.CreatePointerCast(from, to);
    return irbuilder_.CreateBitCast(from, to);
}

Lambda* CodeGen::emit_reinterpret(Lambda* lambda) {
    assert(lambda->num_args() == 3 && "required arguments are missing");
    auto cont = lambda->arg(2)->as_lambda();
    auto type = cont->param(1)->type();
    auto call = emit_bitcast(lambda->arg(1), type);
    emit_result_phi(cont->param(1), call);
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

#ifdef _MSC_VER
    // set dll storage class for MSVC
    if (!entry_ && llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
        if (lambda->empty()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        } else if (lambda->is_external()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }
#endif

    // set linkage
    if (lambda->empty() || lambda->is_external())
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

void CodeGen::emit(int opt, bool debug) {
    if (debug) {
        module_->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
        // Darwin only supports dwarf2
        if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin())
            module_->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
    }
    auto dicompile_unit = dibuilder_.createCompileUnit(llvm::dwarf::DW_LANG_C, world_.name(), llvm::StringRef(), "Impala", opt > 0, llvm::StringRef(), 0);

    Scope::for_each(world_, [&] (const Scope& scope) {
        entry_ = scope.entry();
        assert(entry_->is_returning());
        llvm::Function* fct = emit_function_decl(entry_);

        llvm::DISubprogram disub_program;
        llvm::DIScope* discope = &dicompile_unit;
        if (debug) {
            auto src_file = llvm::sys::path::filename(entry_->loc().begin().filename());
            auto src_dir = llvm::sys::path::parent_path(entry_->loc().begin().filename());
            auto difile = dibuilder_.createFile(src_file, src_dir);
            disub_program = dibuilder_.createFunction(llvm::DIDescriptor(difile), fct->getName(), fct->getName(), difile, entry_->loc().begin().line(),
                                                      dibuilder_.createSubroutineType(difile, dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
                                                      false /* internal linkage */, true /* definition */, entry_->loc().begin().line(),
                                                      llvm::DIDescriptor::FlagPrototyped /* Flags */, opt > 0, fct);
            discope = &disub_program;
        }

        // map params
        const Param* ret_param = nullptr;
        auto arg = fct->arg_begin();
        for (auto param : entry_->params()) {
            if (param->is_mem())
                continue;
            if (param->order() == 0) {
                auto argv = &*arg;
                auto value = map_param(fct, argv, param);
                if (value == argv) {
                    arg->setName(param->unique_name()); // use param
                    params_[param] = arg++;
                } else {
                    params_[param] = value;             // use provided value
                }
            } else {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        BBMap bb2lambda;
        auto schedule = schedule_smart(scope);

        for (const auto& block : schedule) {
            auto lambda = block.lambda();
            // map all bb-like lambdas to llvm bb stubs
            if (lambda->intrinsic() != Intrinsic::EndScope) {
                auto bb = bb2lambda[lambda] = llvm::BasicBlock::Create(context_, lambda->name, fct);

                // create phi node stubs (for all lambdas different from entry)
                if (entry_ != lambda) {
                    for (auto param : lambda->params()) {
                        if (!param->is_mem()) {
                            phis_[param] = llvm::PHINode::Create(convert(param->type()),
                                                                 (unsigned) param->peek().size(), param->name, bb);
                        }
                    }
                }
            }
        }

        auto oldStartBB = fct->begin();
        auto startBB = llvm::BasicBlock::Create(context_, fct->getName() + "_start", fct, oldStartBB);
        irbuilder_.SetInsertPoint(startBB);
        if (debug)
            irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(entry_->loc().begin().line(), entry_->loc().begin().col(), *discope));
        emit_function_start(startBB, entry_);
        irbuilder_.CreateBr(oldStartBB);

        for (auto& block : schedule) {
            auto lambda = block.lambda();
            if (lambda->intrinsic() == Intrinsic::EndScope)
                continue;
            assert(lambda == entry_ || lambda->is_basicblock());
            irbuilder_.SetInsertPoint(bb2lambda[lambda]);

            for (auto primop : block) {
                if (debug)
                    irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(primop->loc().begin().line(), primop->loc().begin().col(), *discope));
                primops_[primop] = emit(primop);
            }

            // terminate bb
            if (debug)
                irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(lambda->jump_loc().begin().line(), lambda->jump_loc().begin().col(), *discope));
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                switch (num_args) {
                    case 0: irbuilder_.CreateRetVoid(); break;
                    case 1:
                        if (lambda->arg(0)->is_mem())
                            irbuilder_.CreateRetVoid();
                        else
                            irbuilder_.CreateRet(lookup(lambda->arg(0)));
                        break;
                    case 2:
                        if (lambda->arg(0)->is_mem()) {
                            irbuilder_.CreateRet(lookup(lambda->arg(1)));
                            break;
                        } else if (lambda->arg(1)->is_mem()) {
                            irbuilder_.CreateRet(lookup(lambda->arg(0)));
                            break;
                        }
                        // FALLTHROUGH
                    default: {
                        Array<llvm::Value*> values(num_args);
                        Array<llvm::Type*> args(num_args);

                        size_t n = 0;
                        for (auto arg : lambda->args()) {
                            if (!arg->is_mem()) {
                                auto val = lookup(arg);
                                values[n] = val;
                                args[n++] = val->getType();
                            }
                        }

                        assert(n == num_args || n+1 == num_args);
                        values.shrink(n);
                        args.shrink(n);
                        llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context_, llvm_ref(args)));

                        for (size_t i = 0; i != n; ++i)
                            agg = irbuilder_.CreateInsertValue(agg, values[i], { unsigned(i) });

                        irbuilder_.CreateRet(agg);
                        break;
                    }
                }
            } else if (lambda->to() == world().branch()) {
                auto cond = lookup(lambda->arg(0));
                auto tbb = bb2lambda[lambda->arg(1)->as_lambda()];
                auto fbb = bb2lambda[lambda->arg(2)->as_lambda()];
                irbuilder_.CreateCondBr(cond, tbb, fbb);
            } else if (lambda->to()->isa<Bottom>()) {
                irbuilder_.CreateUnreachable();
            } else {
                auto to_lambda = lambda->to()->as_lambda();
                if (to_lambda->is_basicblock())         // ordinary jump
                    irbuilder_.CreateBr(bb2lambda[to_lambda]);
                else {
                    if (to_lambda->is_intrinsic()) {
                        auto ret_lambda = emit_intrinsic(lambda);
                        irbuilder_.CreateBr(bb2lambda[ret_lambda]);
                    } else {
                        // put all first-order args into an array
                        std::vector<llvm::Value*> args;
                        Def ret_arg;
                        for (auto arg : lambda->args()) {
                            if (arg->order() == 0) {
                                if (!arg->is_mem())
                                    args.push_back(lookup(arg));
                            } else {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }
                        llvm::CallInst* call = irbuilder_.CreateCall(emit_function_decl(to_lambda), args);
                        // set proper calling convention
                        if (to_lambda->is_external()) {
                            call->setCallingConv(kernel_calling_convention_);
                        } else if (to_lambda->cc() == CC::Device) {
                            call->setCallingConv(device_calling_convention_);
                        } else {
                            call->setCallingConv(function_calling_convention_);
                        }

                        if (ret_arg == ret_param) {     // call + return
                            irbuilder_.CreateRet(call);
                        } else {                        // call + continuation
                            auto succ = ret_arg->as_lambda();
                            const Param* param = nullptr;
                            switch (succ->num_params()) {
                                case 0:
                                    break;
                                case 1:
                                    param = succ->param(0);
                                    irbuilder_.CreateBr(bb2lambda[succ]);
                                    if (!param->is_mem())
                                        emit_result_phi(param, call);
                                    break;
                                case 2:
                                    assert(succ->mem_param() && "no mem_param found for succ");
                                    param = succ->param(0);
                                    param = param->is_mem() ? succ->param(1) : param;
                                    irbuilder_.CreateBr(bb2lambda[succ]);
                                    emit_result_phi(param, call);
                                    break;
                                default: {
                                    assert(succ->param(0)->is_mem());
                                    auto tuple = succ->params().skip_front();

                                    Array<llvm::Value*> extracts(tuple.size());
                                    for (size_t i = 0, e = tuple.size(); i != e; ++i)
                                        extracts[i] = irbuilder_.CreateExtractValue(call, unsigned(i));

                                    irbuilder_.CreateBr(bb2lambda[succ]);
                                    for (size_t i = 0, e = tuple.size(); i != e; ++i)
                                        emit_result_phi(tuple[i], extracts[i]);
                                    break;
                                }
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
    llvm::verifyModule(*module_);
#endif
    optimize(opt);
    if (debug)
        dibuilder_.finalize();

    {
        std::error_code EC;
        auto bc_name = get_binary_output_name(world_.name());
        llvm::raw_fd_ostream out(bc_name, EC, llvm::sys::fs::F_None);
        if (EC)
            throw std::runtime_error("cannot write '" + bc_name + "': " + EC.message());

        llvm::WriteBitcodeToFile(module_.get(), out);
    }

    {
        std::error_code EC;
        auto ll_name = get_output_name(world_.name());
        llvm::raw_fd_ostream out(ll_name, EC, llvm::sys::fs::F_Text);
        if (EC)
            throw std::runtime_error("cannot write '" + ll_name + "': " + EC.message());

        module_->print(out, nullptr);
    }
}

void CodeGen::optimize(int opt) {
    if (opt != 0) {
        llvm::PassManagerBuilder pmbuilder;
        llvm::PassManager pass_manager;
        if (opt == -1) {
            pmbuilder.OptLevel = 2u;
            pmbuilder.SizeLevel = 1;
        } else {
            pmbuilder.OptLevel = (unsigned) opt;
            pmbuilder.SizeLevel = 0u;
        }
        pmbuilder.DisableUnitAtATime = true;
        if (opt == 3) {
            pass_manager.add(llvm::createFunctionInliningPass());
            pass_manager.add(llvm::createAggressiveDCEPass());
        }
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
    auto entry = &irbuilder_.GetInsertBlock()->getParent()->getEntryBlock();
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
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateICmpSGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateICmpSGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateICmpSLT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateICmpSLE(lhs, rhs, name);
                }
            } else if (type->is_type_u() || type->is_bool()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateICmpUGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateICmpUGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateICmpULT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateICmpULE(lhs, rhs, name);
                }
            } else if (type->is_type_pf()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return irbuilder_.CreateFCmpOEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateFCmpONE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateFCmpOGT (lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateFCmpOGE (lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateFCmpOLT (lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateFCmpOLE (lhs, rhs, name);
                }
            } else if (type->is_type_qf()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return irbuilder_.CreateFCmpUEQ(lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateFCmpUNE(lhs, rhs, name);
                    case Cmp_gt: return irbuilder_.CreateFCmpUGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder_.CreateFCmpUGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder_.CreateFCmpULT(lhs, rhs, name);
                    case Cmp_le: return irbuilder_.CreateFCmpULE(lhs, rhs, name);
                }
            } else if (type.isa<PtrType>()) {
                switch (cmp->cmp_kind()) {
                    case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                    default: THORIN_UNREACHABLE;
                }
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();
            bool q = arithop->type()->is_type_q(); // quick? -> nsw/nuw/fast float

            if (type->is_type_f()) {
                switch (arithop->arithop_kind()) {
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

            if (type->is_type_s() || type->is_bool()) {
                switch (arithop->arithop_kind()) {
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
            if (type->is_type_u() || type->is_bool()) {
                switch (arithop->arithop_kind()) {
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
            if (src_type.isa<PtrType>() && dst_type.isa<PtrType>()) {
                return irbuilder_.CreatePointerCast(from, to);
            }
            if (src_type.isa<PtrType>()) {
                assert(dst_type->is_type_i() || dst_type->is_bool());
                return irbuilder_.CreatePtrToInt(from, to);
            }
            if (dst_type.isa<PtrType>()) {
                assert(src_type->is_type_i() || src_type->is_bool());
                return irbuilder_.CreateIntToPtr(from, to);
            }

            auto src = src_type.as<PrimType>();
            auto dst = dst_type.as<PrimType>();

            if (src->is_type_f() && dst->is_type_f()) {
                assert(num_bits(src->primtype_kind()) != num_bits(dst->primtype_kind()));
                return irbuilder_.CreateFPCast(from, to);
            }
            if (src->is_type_f()) {
                if (dst->is_type_s())
                    return irbuilder_.CreateFPToSI(from, to);
                return irbuilder_.CreateFPToUI(from, to);
            }
            if (dst->is_type_f()) {
                if (src->is_type_s())
                    return irbuilder_.CreateSIToFP(from, to);
                return irbuilder_.CreateUIToFP(from, to);
            }

            if (num_bits(src->primtype_kind()) > num_bits(dst->primtype_kind())) {
                if (src->is_type_i() && (dst->is_type_i() || dst->is_bool()))
                    return irbuilder_.CreateTrunc(from, to);
            } else if (num_bits(src->primtype_kind()) < num_bits(dst->primtype_kind())) {
                if ( src->is_type_s()                    && dst->is_type_i()) return irbuilder_.CreateSExt(from, to);
                if ((src->is_type_u() || src->is_bool()) && dst->is_type_i()) return irbuilder_.CreateZExt(from, to);
            }

            assert(false && "unsupported cast");
        }

        if (conv->isa<Bitcast>())
            return emit_bitcast(conv->from(), dst_type);
    }

    if (auto select = def->isa<Select>()) {
        if (def->type().isa<FnType>())
            return nullptr;

        llvm::Value* cond = lookup(select->cond());
        llvm::Value* tval = lookup(select->tval());
        llvm::Value* fval = lookup(select->fval());
        return irbuilder_.CreateSelect(cond, tval, fval);
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
        WLOG("slow: alloca and loads/stores needed for definite array '%' at '%'", def, def->loc());
        auto alloca = emit_alloca(type, array->name);

        u64 i = 0;
        llvm::Value* args[2] = { irbuilder_.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = irbuilder_.getInt64(i++);
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args, op->name);
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
            for (size_t i = 0, e = agg->ops().size(); i != e; ++i)
                llvm_agg = irbuilder_.CreateInsertElement(llvm_agg, lookup(agg->op(i)), irbuilder_.getInt32(i));
        } else {
            for (size_t i = 0, e = agg->ops().size(); i != e; ++i)
                llvm_agg = irbuilder_.CreateInsertValue(llvm_agg, lookup(agg->op(i)), { unsigned(i) });
        }

        return llvm_agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto llvm_agg = lookup(aggop->agg());
        auto llvm_idx = lookup(aggop->index());
        auto copy_to_alloca = [&] () {
            WLOG("slow: alloca and loads/stores needed for aggregate '%' at '%'", def, def->loc());
            auto alloca = emit_alloca(llvm_agg->getType(), aggop->name);
            irbuilder_.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { irbuilder_.getInt64(0), llvm_idx };
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args);
            return std::make_pair(alloca, gep);
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (auto memop = extract->agg()->isa<MemOp>())
                return lookup(memop);

            if (aggop->agg()->type().isa<ArrayType>())
                return irbuilder_.CreateLoad(copy_to_alloca().second);

            if (extract->agg()->type().isa<VectorType>())
                return irbuilder_.CreateExtractElement(llvm_agg, llvm_idx);
            // tuple/struct
            return irbuilder_.CreateExtractValue(llvm_agg, {aggop->index()->primlit_value<unsigned>()});
        }

        auto insert = def->as<Insert>();
        auto value = lookup(insert->value());

        if (insert->agg()->type().isa<ArrayType>()) {
            auto p = copy_to_alloca();
            irbuilder_.CreateStore(lookup(aggop->as<Insert>()->value()), p.second);
            return irbuilder_.CreateLoad(p.first);
        }
        if (insert->agg()->type().isa<VectorType>())
            return irbuilder_.CreateInsertElement(llvm_agg, lookup(aggop->as<Insert>()->value()), llvm_idx);
        // tuple/struct
        return irbuilder_.CreateInsertValue(llvm_agg, value, {aggop->index()->primlit_value<unsigned>()});
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* llvm_type = convert(primlit->type());
        Box box = primlit->value();

        switch (primlit->primtype_kind()) {
            case PrimType_bool:                     return irbuilder_. getInt1(box.get_bool());
            case PrimType_ps8:  case PrimType_qs8:  return irbuilder_. getInt8(box. get_s8());
            case PrimType_pu8:  case PrimType_qu8:  return irbuilder_. getInt8(box. get_u8());
            case PrimType_ps16: case PrimType_qs16: return irbuilder_.getInt16(box.get_s16());
            case PrimType_pu16: case PrimType_qu16: return irbuilder_.getInt16(box.get_u16());
            case PrimType_ps32: case PrimType_qs32: return irbuilder_.getInt32(box.get_s32());
            case PrimType_pu32: case PrimType_qu32: return irbuilder_.getInt32(box.get_u32());
            case PrimType_ps64: case PrimType_qs64: return irbuilder_.getInt64(box.get_s64());
            case PrimType_pu64: case PrimType_qu64: return irbuilder_.getInt64(box.get_u64());
            case PrimType_pf32: case PrimType_qf32: return llvm::ConstantFP::get(llvm_type, box.get_f32());
            case PrimType_pf64: case PrimType_qf64: return llvm::ConstantFP::get(llvm_type, box.get_f64());
        }
    }

    if (auto bottom = def->isa<Bottom>())
        return llvm::UndefValue::get(convert(bottom->type()));

    if (auto alloc = def->isa<Alloc>()) { // TODO factor this code
        // TODO do this only once
        auto llvm_malloc = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
                    get_alloc_name(), irbuilder_.getInt8PtrTy(), irbuilder_.getInt32Ty(), irbuilder_.getInt64Ty(), nullptr));
        llvm_malloc->addAttribute(llvm::AttributeSet::ReturnIndex, llvm::Attribute::NoAlias);
        auto alloced_type = convert(alloc->alloced_type());
        llvm::CallInst* void_ptr;
        auto layout = module_->getDataLayout();
        if (auto array = alloc->alloced_type()->is_indefinite()) {
            auto size = irbuilder_.CreateAdd(
                    irbuilder_.getInt64(layout->getTypeAllocSize(alloced_type)),
                    irbuilder_.CreateMul(irbuilder_.CreateIntCast(lookup(alloc->extra()), irbuilder_.getInt64Ty(), false),
                                         irbuilder_.getInt64(layout->getTypeAllocSize(convert(array->elem_type())))));
            llvm::Value* malloc_args[] = {
                irbuilder_.getInt32(0),
                size
            };
            void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
        } else {
            llvm::Value* malloc_args[] = {
                irbuilder_.getInt32(0),
                irbuilder_.getInt64(layout->getTypeAllocSize(alloced_type))
            };
            void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
        }

        return irbuilder_.CreatePointerCast(void_ptr, convert(alloc->out_ptr_type()));
    }

    if (auto load = def->isa<Load>())    return emit_load(load);
    if (auto store = def->isa<Store>())  return emit_store(store);
    if (auto lea = def->isa<LEA>())      return emit_lea(lea);
    if (def->isa<Enter>())               return nullptr;

    if (auto slot = def->isa<Slot>())
        return irbuilder_.CreateAlloca(convert(slot->type().as<PtrType>()->referenced_type()), 0, slot->unique_name());

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(convert(vector->type()));
        for (size_t i = 0, e = vector->size(); i != e; ++i)
            vec = irbuilder_.CreateInsertElement(vec, lookup(vector->op(i)), lookup(world_.literal_pu32(i, vector->loc())));

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

llvm::Value* CodeGen::emit_load(const Load* load) {
    return irbuilder_.CreateLoad(lookup(load->ptr()));
}

llvm::Value* CodeGen::emit_store(const Store* store) {
    return irbuilder_.CreateStore(lookup(store->val()), lookup(store->ptr()));
}

llvm::Value* CodeGen::emit_lea(const LEA* lea) {
    if (lea->ptr_referenced_type().isa<TupleType>() || lea->ptr_referenced_type().isa<StructAppType>())
        return irbuilder_.CreateStructGEP(lookup(lea->ptr()), lea->index()->primlit_value<u32>());

    assert(lea->ptr_referenced_type().isa<ArrayType>());
    llvm::Value* args[2] = { irbuilder_.getInt64(0), lookup(lea->index()) };
    return irbuilder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
}

llvm::Type* CodeGen::convert(Type type) {
    if (auto ltype = thorin::find(types_, *type.unify()))
        return ltype;

    assert(!type.isa<MemType>());
    llvm::Type* llvm_type;
    switch (type->kind()) {
        case PrimType_bool:                                                             llvm_type = irbuilder_. getInt1Ty(); break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  llvm_type = irbuilder_. getInt8Ty(); break;
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: llvm_type = irbuilder_.getInt16Ty(); break;
        case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: llvm_type = irbuilder_.getInt32Ty(); break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: llvm_type = irbuilder_.getInt64Ty(); break;
        case PrimType_pf32: case PrimType_qf32:                                         llvm_type = irbuilder_.getFloatTy(); break;
        case PrimType_pf64: case PrimType_qf64:                                         llvm_type = irbuilder_.getDoubleTy();break;
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

llvm::GlobalVariable* CodeGen::emit_global_variable(llvm::Type* type, const std::string& name, unsigned addr_space) {
    return new llvm::GlobalVariable(*module_, type, false,
            llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(type), name,
            nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
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

    if (!cuda.empty() || !nvvm.empty() || !spir.empty() || !opencl.empty())
        world.cleanup();

    CPUCodeGen(world).emit(opt, debug);
    if (!cuda.  empty()) CUDACodeGen(cuda).emit(/*opt,*/ debug);
    if (!nvvm.  empty()) NVVMCodeGen(nvvm).emit(opt, debug);
    if (!spir.  empty()) SPIRCodeGen(spir).emit(opt, debug);
    if (!opencl.empty()) OpenCLCodeGen(opencl).emit(/*opt,*/ debug);
}

//------------------------------------------------------------------------------

}
