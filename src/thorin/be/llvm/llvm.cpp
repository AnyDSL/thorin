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
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>

#include "thorin/config.h"
#if THORIN_ENABLE_RV
#include <rv/rv.h>
#endif

#include "thorin/def.h"
#include "thorin/match.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/llvm/amdgpu.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/hls.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/opencl.h"
#include "thorin/pass/optimize.h"
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

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

Lam* CodeGen::emit_intrinsic(Lam* lam) {
    auto callee = lam->app()->callee()->as_nominal<Lam>();
    switch (callee->intrinsic()) {
        case Lam::Intrinsic::Atomic:    return emit_atomic(lam);
        case Lam::Intrinsic::CmpXchg:   return emit_cmpxchg(lam);
        case Lam::Intrinsic::Reserve:   return emit_reserve(lam);
        case Lam::Intrinsic::CUDA:      return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM,   ".cu",     lam);
        case Lam::Intrinsic::NVVM:      return runtime_->emit_host_code(*this, Runtime::CUDA_PLATFORM,   ".nvvm",   lam);
        case Lam::Intrinsic::OpenCL:    return runtime_->emit_host_code(*this, Runtime::OPENCL_PLATFORM, ".cl",     lam);
        case Lam::Intrinsic::AMDGPU:    return runtime_->emit_host_code(*this, Runtime::HSA_PLATFORM,    ".amdgpu", lam);
        case Lam::Intrinsic::HLS:       return emit_hls(lam);
        case Lam::Intrinsic::Parallel:  return emit_parallel(lam);
        case Lam::Intrinsic::Spawn:     return emit_spawn(lam);
        case Lam::Intrinsic::Sync:      return emit_sync(lam);
#if THORIN_ENABLE_RV
        case Lam::Intrinsic::Vectorize: return emit_vectorize_lam(lam);
#else
        case Lam::Intrinsic::Vectorize: throw std::runtime_error("rebuild with RV support");
#endif
        default: THORIN_UNREACHABLE;
    }
}

Lam* CodeGen::emit_hls(Lam* lam) {
    std::vector<llvm::Value*> args(lam->app()->num_args()-3);
    Lam* ret = nullptr;
    for (size_t i = 2, j = 0; i < lam->app()->num_args(); ++i) {
        if (auto l = lam->app()->arg(i)->isa_nominal<Lam>()) {
            ret = l;
            continue;
        }
        args[j++] = emit(lam->app()->arg(i));
    }
    auto callee = lam->app()->arg(1)->as<Global>()->init()->as_nominal<Lam>();
    callee->make_external();
    irbuilder_.CreateCall(emit_function_decl(callee), args);
    assert(ret);
    return ret;
}

void CodeGen::emit_result_phi(const Def* param, llvm::Value* value) {
    phis_.lookup(param).value()->addIncoming(value, irbuilder_.GetInsertBlock());
}

// TODO remove once we got rid of signedness
static u64 num_bits(const Def* type) {
    if (auto arg = isa<Tag::Int >(type)) return as_lit<u64>(arg);
    if (auto arg = isa<Tag::Real>(type)) return as_lit<u64>(arg);
    THORIN_UNREACHABLE;
}

Lam* CodeGen::emit_atomic(Lam* lam) {
    assert(lam->app()->num_args() == 5 && "required arguments are missing");
    if (!isa<Tag::Int>(lam->app()->arg(3)->type()))
        EDEF(lam->app()->arg(3), "atomic only supported for integer types");
    // atomic tag: Xchg Add Sub And Nand Or Xor Max Min
    u32 tag = as_lit<u32>(lam->app()->arg(1));
    auto ptr = lookup(lam->app()->arg(2));
    auto val = lookup(lam->app()->arg(3));
    assert(int(llvm::AtomicRMWInst::BinOp::Xchg) <= int(tag) && int(tag) <= int(llvm::AtomicRMWInst::BinOp::UMin) && "unsupported atomic");
    auto binop = (llvm::AtomicRMWInst::BinOp)tag;
    auto l = lam->app()->arg(4)->as_nominal<Lam>();
    auto call = irbuilder_.CreateAtomicRMW(binop, ptr, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SyncScope::System);
    emit_result_phi(l->param(1), call);
    return l;
}

Lam* CodeGen::emit_cmpxchg(Lam* lam) {
    assert(lam->app()->num_args() == 5 && "required arguments are missing");
    if (!isa<Tag::Int>(lam->app()->arg(3)->type()))
        EDEF(lam->app()->arg(3), "cmpxchg only supported for integer types");
    auto ptr  = lookup(lam->app()->arg(1));
    auto cmp  = lookup(lam->app()->arg(2));
    auto val  = lookup(lam->app()->arg(3));
    auto l = lam->app()->arg(4)->as_nominal<Lam>();
    auto call = irbuilder_.CreateAtomicCmpXchg(ptr, cmp, val, llvm::AtomicOrdering::SequentiallyConsistent, llvm::AtomicOrdering::SequentiallyConsistent, llvm::SyncScope::System);
    emit_result_phi(l->param(1), irbuilder_.CreateExtractValue(call, 0));
    emit_result_phi(l->param(2), irbuilder_.CreateExtractValue(call, 1));
    return l;
}

Lam* CodeGen::emit_reserve(const Lam* lam) {
    EDEF(lam->app()->debug(), "reserve_shared: only allowed in device code");
    THORIN_UNREACHABLE;
}

#if 0
Lam* CodeGen::emit_reserve_shared(const Lam* lam, bool init_undef) {
    assert(lam->app()->num_args() == 3 && "required arguments are missing");
    if (!lam->app()->arg(1)->isa<Lit>())
        EDEF(lam->app()->arg(1), "reserve_shared: couldn't extract memory size");
    auto num_elems = as_lit<ps32>(lam->app()->arg(1));
    auto l = lam->app()->arg(2)->as_nominal<Lam>();
    auto type = convert(lam->param(1)->type());
    // construct array type
    auto elem_type = l->param(1)->type()->as<Ptr>()->pointee()->as<Variadic>()->body();
    auto smem_type = this->convert(lam->world().variadic(num_elems, elem_type));
    auto name = lam->unique_name();
    // NVVM doesn't allow '.' in global identifier
    std::replace(name.begin(), name.end(), '.', '_');
    auto global = emit_global_variable(smem_type, name, 3, init_undef);
    auto call = irbuilder_.CreatePointerCast(global, type);
    emit_result_phi(l->param(1), call);
    return l;
}
#endif

llvm::Value* CodeGen::emit_bitcast(const Def* val, const Def* dst_type) {
    auto from = lookup(val);
    auto src_type = val->type();
    auto to = convert(dst_type);
    if (from->getType()->isAggregateType() || to->isAggregateType())
        EDEF(val, "bitcast from or to aggregate types not allowed: bitcast from '{}' to '{}'", src_type, dst_type);
    if (src_type->isa<Ptr>() && dst_type->isa<Ptr>())
        return irbuilder_.CreatePointerCast(from, to);
    return irbuilder_.CreateBitCast(from, to);
}

llvm::FunctionType* CodeGen::convert_fn_type(Lam* lam) {
    return llvm::cast<llvm::FunctionType>(convert(lam->type()));
}

llvm::Function* CodeGen::emit_function_decl(Lam* lam) {
    if (auto f = fcts_.lookup(lam)) return *f;

    std::string name = (lam->is_external() || !lam->is_set()) ? lam->name() : lam->unique_name();
    auto f = llvm::cast<llvm::Function>(module_->getOrInsertFunction(name, convert_fn_type(lam)));

#ifdef _MSC_VER
    // set dll storage class for MSVC
    if (!entry_ && llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
        if (lam->empty()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        } else if (lam->is_external()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }
#endif

    // set linkage
    if (!lam->is_set() || lam->is_external())
        f->setLinkage(llvm::Function::ExternalLinkage);
    else
        f->setLinkage(llvm::Function::InternalLinkage);

    // set calling convention
    if (lam->is_external()) {
        f->setCallingConv(kernel_calling_convention_);
        emit_function_decl_hook(lam, f);
    } else {
        if (lam->cc() == Lam::CC::Device)
            f->setCallingConv(device_calling_convention_);
        else
            f->setCallingConv(function_calling_convention_);
    }

    return fcts_[lam] = f;
}

std::unique_ptr<llvm::Module>& CodeGen::emit(int opt, bool debug) {
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
            auto src_file = llvm::sys::path::filename(entry_->filename());
            auto src_dir = llvm::sys::path::parent_path(entry_->filename());
            auto difile = dibuilder_.createFile(src_file, src_dir);
            disub_program = dibuilder_.createFunction(discope, fct->getName(), fct->getName(), difile, entry_->front_line(),
                                                      dibuilder_.createSubroutineType(dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
                                                      entry_->front_line(),
                                                      llvm::DINode::FlagPrototyped,
                                                      llvm::DISubprogram::SPFlagDefinition | (opt > 0 ? llvm::DISubprogram::SPFlagOptimized : llvm::DISubprogram::SPFlagZero));
            fct->setSubprogram(disub_program);
            discope = disub_program;
        }

        // map params
        const Def* ret_param = nullptr;
        auto arg = fct->arg_begin();
        for (auto param : entry_->params()) {
            if (param->type()->isa<Mem>() || is_unit(param)) {
                params_[param] = nullptr;
            } else if (param->type()->order() == 0) {
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
                params_[param] = nullptr;
            }
        }
        assert(ret_param);

        BBMap bb2lam;
        Schedule schedule(scope);

        for (const auto& block : schedule) {
            auto lam = block.lam();
            // map all bb-like lams to llvm bb stubs
            if (lam->intrinsic() != Lam::Intrinsic::EndScope) {
                auto bb = bb2lam[lam] = llvm::BasicBlock::Create(context_, lam->name(), fct);

                // create phi node stubs (for all lams different from entry)
                if (entry_ != lam) {
                    for (auto param : lam->params()) {
                        auto phi = (param->type()->isa<Mem>() || is_unit(param))
                                 ? nullptr
                                 : llvm::PHINode::Create(convert(param->type()), (unsigned) peek(param).size(), param->name(), bb);
                        phis_[param] = phi;
                    }
                }
            }
        }

        auto oldStartBB = fct->begin();
        auto startBB = llvm::BasicBlock::Create(context_, fct->getName() + "_start", fct, &*oldStartBB);
        irbuilder_.SetInsertPoint(startBB);
        if (debug)
            irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(entry_->front_line(), entry_->front_col(), discope));
        emit_function_start(startBB, entry_);
        irbuilder_.CreateBr(&*oldStartBB);

        for (auto& block : schedule) {
            auto lam = block.lam();
            if (lam->intrinsic() == Lam::Intrinsic::EndScope)
                continue;
            assert(lam == entry_ || lam->is_basicblock());
            irbuilder_.SetInsertPoint(bb2lam[lam]);

            for (auto def : block) {
                if (debug)
                    irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(def->front_line(), def->front_col(), discope));

                if (def->isa<Param>()) continue;
                if (def->isa<App>()) continue;
                auto i = phis_.  find(def);
                if (i != phis_.  end()) continue;
                auto j = params_.find(def);
                if (j != params_.end()) continue;

                if (is_tuple_arg_of_app(def)) continue;

#if 0
                // ignore tuple arguments for lams
                if (auto tuple = def->isa<Tuple>()) {
                    bool ignore = false;
                    for (auto use : tuple->uses()) {
                        ignore |= use->isa<Lam>() != nullptr;
                    }

                    if (ignore) continue;
                }

                if (def->type()->order() >= 1) {
                    // ignore higher-order defs which stem from a branch/match intrinsic
                    if (is_arg_of_app(def)) continue;
                    THORIN_UNREACHABLE;
                }
#endif

                if (auto llvm_value = emit(def))
                    defs_[def] = llvm_value;
            }

            // terminate bb
            if (debug)
                irbuilder_.SetCurrentDebugLocation(llvm::DebugLoc::get(lam->app()->front_line(), lam->app()->front_col(), discope));
            if (lam->app()->callee() == ret_param) { // return
                size_t num_args = lam->app()->num_args();
                if (num_args == 0) irbuilder_.CreateRetVoid();
                else {
                    Array<llvm::Value*> values(num_args);
                    Array<llvm::Type*> args(num_args);

                    size_t n = 0;
                    for (auto arg : lam->app()->args()) {
                        if (!arg->type()->isa<Mem>() && !is_unit(arg)) {
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
            } else if (auto select = isa<Tag::Select>(lam->app()->callee())) {
                auto [cond, a, b] = split<3>(select);
                auto tbb = bb2lam[a->as_nominal<Lam>()];
                auto fbb = bb2lam[b->as_nominal<Lam>()];
                irbuilder_.CreateCondBr(lookup(cond), tbb, fbb);
            } else if (lam->app()->callee()->isa<Lam>() &&
                       lam->app()->callee()->as<Lam>()->intrinsic() == Lam::Intrinsic::Match) {
                auto val = lookup(lam->app()->arg(0));
                auto otherwise_bb = bb2lam[lam->app()->arg(1)->as_nominal<Lam>()];
                auto match = irbuilder_.CreateSwitch(val, otherwise_bb, lam->app()->num_args() - 2);
                for (size_t i = 2; i < lam->app()->num_args(); i++) {
                    auto arg = lam->app()->arg(i)->as<Tuple>();
                    auto case_const = llvm::cast<llvm::ConstantInt>(lookup(arg->op(0)));
                    auto case_bb    = bb2lam[arg->op(1)->as_nominal<Lam>()];
                    match->addCase(case_const, case_bb);
                }
            } else if (lam->app()->callee()->isa<Bot>()) {
                irbuilder_.CreateUnreachable();
            } else {
                auto callee = lam->app()->callee();
                bool terminated = false;
                if (auto callee_lam = callee->isa_nominal<Lam>()) {
                    if (callee_lam->is_basicblock()) {
                        // ordinary jump
                        irbuilder_.CreateBr(bb2lam[callee_lam]);
                        terminated = true;
                    } else if (callee_lam->is_intrinsic()) {
                        // intrinsic call
                        auto ret_lam = emit_intrinsic(lam);
                        irbuilder_.CreateBr(bb2lam[ret_lam]);
                        terminated = true;
                    }
                }

                // function/closure call
                if (!terminated) {
                    // put all first-order args into an array
                    std::vector<llvm::Value*> args;
                    const Def* ret_arg = nullptr;
                    for (auto arg : lam->app()->args()) {
                        if (arg->type()->order() == 0) {
                            if (!arg->type()->isa<Mem>() && !is_unit(arg))
                                args.push_back(lookup(arg));
                        } else {
                            assert(!ret_arg);
                            ret_arg = arg;
                        }
                    }

                    llvm::CallInst* call = nullptr;
                    if (auto callee_lam = callee->isa_nominal<Lam>()) {
                        call = irbuilder_.CreateCall(emit_function_decl(callee_lam), args);
                        if (callee_lam->is_external())
                            call->setCallingConv(kernel_calling_convention_);
                        else if (callee_lam->cc() == Lam::CC::Device)
                            call->setCallingConv(device_calling_convention_);
                        else
                            call->setCallingConv(function_calling_convention_);
                    } else {
                        // must be a closure
                        auto closure = lookup(callee);
                        args.push_back(irbuilder_.CreateExtractValue(closure, 1));
                        call = irbuilder_.CreateCall(irbuilder_.CreateExtractValue(closure, 0), args);
                    }

                    // must be call + lam --- call + return has been removed by codegen_prepare
                    auto succ = ret_arg->as_nominal<Lam>();

                    size_t n = 0;
                    const Def* last_param = nullptr;
                    for (auto param : succ->params()) {
                        if (param->type()->isa<Mem>() || is_unit(param))
                            continue;
                        last_param = param;
                        ++n;
                    }

                    if (n == 0) {
                        irbuilder_.CreateBr(bb2lam[succ]);
                    } else if (n == 1) {
                        irbuilder_.CreateBr(bb2lam[succ]);
                        emit_result_phi(last_param, call);
                    } else {
                        Array<llvm::Value*> extracts(n);
                        for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                            auto param = succ->param(i);
                            if (param->type()->isa<Mem>() || is_unit(param))
                                continue;
                            extracts[j] = irbuilder_.CreateExtractValue(call, unsigned(j));
                            j++;
                        }

                        irbuilder_.CreateBr(bb2lam[succ]);

                        for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                            auto param = succ->param(i);
                            if (param->type()->isa<Mem>() || is_unit(param))
                                continue;
                            emit_result_phi(param, extracts[j]);
                            j++;
                        }
                    }
                }
            }
        }

        // add missing arguments to phis_
        for (const auto& p : phis_) {
            if (auto phi = p.second) {
                auto param = p.first;
                for (auto&& p : peek(param))
                    phi->addIncoming(lookup(p.def()), bb2lam[p.from()]);
            }
        }

        params_.clear();
        phis_.clear();
        defs_.clear();
    });

    if (debug)
        dibuilder_.finalize();

#if THORIN_ENABLE_RV
    // emit vectorized code
    for (const auto& tuple : vec_todo_)
        emit_vectorize(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
    vec_todo_.clear();

    rv::lowerIntrinsics(*module_);
#endif

#if THORIN_ENABLE_CHECKS
    llvm::verifyModule(*module_);
#endif
    optimize(opt);

    return module_;
}

void CodeGen::emit(std::ostream& stream, int opt, bool debug) {
    llvm::raw_os_ostream llvm_stream(stream);
    emit(opt, debug)->print(llvm_stream, nullptr);
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
    auto i = phis_.  find(def);
    if (i != phis_.  end()) return i->second;
    auto j = params_.find(def);
    if (j != params_.end()) return j->second;

    if (auto lam = def->isa_nominal<Lam>())
        return emit_function_decl(lam);

    if (auto res = defs_.lookup(def))
        return *res;
    else {
        // we emit all Thorin constants in the entry block, since they are not part of the schedule
        if (is_const(def)) {
            auto bb = irbuilder_.GetInsertBlock();
            auto fn = bb->getParent();
            auto& entry = fn->getEntryBlock();

            auto dbg = irbuilder_.getCurrentDebugLocation();
            auto ip = irbuilder_.saveAndClearIP();
            irbuilder_.SetInsertPoint(&entry, entry.begin());
            auto llvm_value = emit(def);
            irbuilder_.restoreIP(ip);
            irbuilder_.SetCurrentDebugLocation(dbg);
            return defs_[def] = llvm_value;
        }

        auto llvm_value = emit(def);
        return defs_[def] = llvm_value;
    }
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::Type* type, const std::string& name) {
    auto entry = &irbuilder_.GetInsertBlock()->getParent()->getEntryBlock();
    auto layout = llvm::DataLayout(module_->getDataLayout());
    llvm::AllocaInst* alloca;
    if (entry->empty())
        alloca = new llvm::AllocaInst(type, layout.getAllocaAddrSpace(), nullptr, name, entry);
    else
        alloca = new llvm::AllocaInst(type, layout.getAllocaAddrSpace(), nullptr, name, entry->getFirstNonPHIOrDbg());
    return alloca;
}

llvm::Value* CodeGen::emit_alloc(const Def* type) {
    auto llvm_malloc = runtime_->get(get_alloc_name().c_str());
    auto alloced_type = convert(type);
    llvm::CallInst* void_ptr;
    auto layout = module_->getDataLayout();
    if (auto variadic = type->isa<Variadic>()) {
        auto num = lookup(variadic->arity());
        auto size = irbuilder_.CreateAdd(
                irbuilder_.getInt64(layout.getTypeAllocSize(alloced_type)),
                irbuilder_.CreateMul(irbuilder_.CreateIntCast(num, irbuilder_.getInt64Ty(), false),
                                     irbuilder_.getInt64(layout.getTypeAllocSize(convert(variadic->body())))));
        llvm::Value* malloc_args[] = { irbuilder_.getInt32(0), size };
        void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
    } else {
        llvm::Value* malloc_args[] = { irbuilder_.getInt32(0), irbuilder_.getInt64(layout.getTypeAllocSize(alloced_type)) };
        void_ptr = irbuilder_.CreateCall(llvm_malloc, malloc_args);
    }

    return irbuilder_.CreatePointerCast(void_ptr, llvm::PointerType::get(alloced_type, 0));
}

llvm::Value* CodeGen::emit(const Def* def) {
    if (auto cmp = isa<Tag::ICmp>(def)) {
        auto [a, b] = split<2>(cmp);
        auto x = lookup(a);
        auto y = lookup(b);
        auto name = def->name();
        switch (ICmp(std::get<0>(get_axiom(def))->flags())) {
            case ICmp::e:   return irbuilder_.CreateICmpEQ (x, y, name);
            case ICmp::ne:  return irbuilder_.CreateICmpNE (x, y, name);
            case ICmp::sg:  return irbuilder_.CreateICmpSGT(x, y, name);
            case ICmp::sge: return irbuilder_.CreateICmpSGE(x, y, name);
            case ICmp::sl:  return irbuilder_.CreateICmpSLT(x, y, name);
            case ICmp::sle: return irbuilder_.CreateICmpSLE(x, y, name);
            case ICmp::ug:  return irbuilder_.CreateICmpUGT(x, y, name);
            case ICmp::uge: return irbuilder_.CreateICmpUGE(x, y, name);
            case ICmp::ul:  return irbuilder_.CreateICmpULT(x, y, name);
            case ICmp::ule: return irbuilder_.CreateICmpULE(x, y, name);
            default: THORIN_UNREACHABLE;
        }
    } else if (auto cmp = isa<Tag::RCmp>(def)) {
        auto [a, b] = split<2>(cmp);
        auto x = lookup(a);
        auto y = lookup(b);
        auto name = def->name();
        switch (RCmp(std::get<0>(get_axiom(def))->flags())) {
            case RCmp::  e: return irbuilder_.CreateFCmpOEQ(x, y, name);
            case RCmp::  l: return irbuilder_.CreateFCmpOLT(x, y, name);
            case RCmp:: le: return irbuilder_.CreateFCmpOLE(x, y, name);
            case RCmp::  g: return irbuilder_.CreateFCmpOGT(x, y, name);
            case RCmp:: ge: return irbuilder_.CreateFCmpOGE(x, y, name);
            case RCmp:: ne: return irbuilder_.CreateFCmpONE(x, y, name);
            case RCmp::  o: return irbuilder_.CreateFCmpORD(x, y, name);
            case RCmp::  u: return irbuilder_.CreateFCmpUNO(x, y, name);
            case RCmp:: ue: return irbuilder_.CreateFCmpUEQ(x, y, name);
            case RCmp:: ul: return irbuilder_.CreateFCmpULT(x, y, name);
            case RCmp::ule: return irbuilder_.CreateFCmpULE(x, y, name);
            case RCmp:: ug: return irbuilder_.CreateFCmpUGT(x, y, name);
            case RCmp::uge: return irbuilder_.CreateFCmpUGE(x, y, name);
            case RCmp::une: return irbuilder_.CreateFCmpUNE(x, y, name);
            default: THORIN_UNREACHABLE;
        }
        //} else if (type->isa<Ptr>()) {
            //switch (cmp->cmp_tag()) {
                //case Cmp_eq: return irbuilder_.CreateICmpEQ (lhs, rhs, name);
                //case Cmp_ne: return irbuilder_.CreateICmpNE (lhs, rhs, name);
                //default: THORIN_UNREACHABLE;
            //}
        //}
    }

    if (auto arithop = def->isa<ArithOp>()) {
        auto lhs = lookup(arithop->lhs());
        auto rhs = lookup(arithop->rhs());
        auto name = arithop->name();
        auto type = arithop->type();

        if (type->isa<Real>()) {
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

        if (type->isa<Sint>() || type->isa<Bool>()) {
            bool q = arithop->type()->isa<Sint>() ? arithop->type()->as<Sint>()->is_quick() : false; // quick? -> nsw/nuw
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

        if (type->isa<Uint>() || type->isa<Bool>()) {
            bool q = arithop->type()->isa<Uint>() ? arithop->type()->as<Uint>()->is_quick() : false; // quick? -> nsw/nuw
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

    if (auto cast = def->isa<Cast>()) {
        auto from = lookup(cast->from());
        auto src = cast->from()->type();
        auto dst = cast->type();
        auto to = convert(dst);

        if (is_arity(dst)) return from;
        if (auto variant_type = src->isa<VariantType>()) {
            auto bits = compute_variant_bits(variant_type);
            if (bits != 0) {
                auto value_bits = compute_variant_op_bits(dst);
                auto trunc = irbuilder_.CreateTrunc(from, irbuilder_.getIntNTy(value_bits));
                return irbuilder_.CreateBitOrPointerCast(trunc, to);
            } else {
                WDEF(def, "slow: alloca and loads/stores needed for variant cast '{}'", def);
                auto ptr_type = llvm::PointerType::get(to, 0);
                return create_tmp_alloca(from->getType(), [&] (auto alloca) {
                    auto casted_ptr = irbuilder_.CreateBitCast(alloca, ptr_type);
                    irbuilder_.CreateStore(from, alloca);
                    return irbuilder_.CreateLoad(casted_ptr);
                });
            }
        }

        if (src->isa<Ptr>() && dst->isa<Ptr>()) {
            return irbuilder_.CreatePointerCast(from, to);
        }
        if (src->isa<Ptr>()) {
            assert(is_type_i(dst) || dst->isa<Bool>());
            return irbuilder_.CreatePtrToInt(from, to);
        }
        if (dst->isa<Ptr>()) {
            assert(is_type_i(src) || src->isa<Bool>());
            return irbuilder_.CreateIntToPtr(from, to);
        }
        if (src->isa<Real>() && dst->isa<Real>()) {
            assert(src->as<Real>()->lit_num_bits() != dst->as<Real>()->lit_num_bits());
            return irbuilder_.CreateFPCast(from, to);
        }
        if (src->isa<Real>()) {
            if (dst->isa<Sint>())
                return irbuilder_.CreateFPToSI(from, to);
            return irbuilder_.CreateFPToUI(from, to);
        }
        if (dst->isa<Real>()) {
            if (src->isa<Sint>())
                return irbuilder_.CreateSIToFP(from, to);
            return irbuilder_.CreateUIToFP(from, to);
        }

        if (num_bits(src) > num_bits(dst)) {
            if (is_type_i(src) && (is_type_i(dst) || dst->isa<Bool>()))
                return irbuilder_.CreateTrunc(from, to);
        } else if (num_bits(src) < num_bits(dst)) {
            if (src->isa<Sint>()                       && is_type_i(dst)) return irbuilder_.CreateSExt(from, to);
            if ((src->isa<Uint>() || src->isa<Bool>()) && is_type_i(dst)) return irbuilder_.CreateZExt(from, to);
        } else if (is_type_i(src) && is_type_i(dst)) {
            assert(num_bits(src) == num_bits(dst));
            return from;
        }

        assert(false && "unsupported cast");
    }

    if (auto bitcast = def->isa<Bitcast>())
        return emit_bitcast(bitcast->from(), bitcast->type());

    if (auto select = def->isa<Select>()) {
        if (def->type()->isa<Pi>())
            return nullptr;

        auto cond = lookup(select->cond());
        auto tval = lookup(select->tval());
        auto fval = lookup(select->fval());
        return irbuilder_.CreateSelect(cond, tval, fval);
    }

    if (auto size_of = def->isa<SizeOf>()) {
        auto type = convert(size_of->of());
        auto layout = llvm::DataLayout(module_->getDataLayout());
        return irbuilder_.getInt32(layout.getTypeAllocSize(type));
    }

#if 0
    if (auto array = def->isa<DefiniteArray>()) {
        auto type = llvm::cast<llvm::ArrayType>(convert(array->type()));
        if (is_const(array)) {
            size_t size = array->num_ops();
            Array<llvm::Constant*> vals(size);
            for (size_t i = 0; i != size; ++i)
                vals[i] = llvm::cast<llvm::Constant>(emit(array->op(i)));
            return llvm::ConstantArray::get(type, llvm_ref(vals));
        }
        WDEF(def, "slow: alloca and loads/stores needed for definite array '{}'", def);
        auto alloca = emit_alloca(type, array->name().str());

        u64 i = 0;
        llvm::Value* args[2] = { irbuilder_.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = irbuilder_.getInt64(i++);
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args, op->name().c_str());
            irbuilder_.CreateStore(lookup(op), gep);
        }

        return irbuilder_.CreateLoad(alloca);
    }

    if (auto array = def->isa<IndefiniteArray>())
        return llvm::UndefValue::get(convert(array->type()));
#endif

    if (auto tuple = def->isa<Tuple>()) {
        llvm::Value* llvm_agg = llvm::UndefValue::get(convert(tuple->type()));

        for (size_t i = 0, e = tuple->num_ops(); i != e; ++i)
            llvm_agg = irbuilder_.CreateInsertValue(llvm_agg, lookup(tuple->op(i)), { unsigned(i) });

        return llvm_agg;
    }

    if (auto pack = def->isa<Pack>()) {
        auto llvm_type = convert(pack->type());
        if (auto lit = isa_lit<u64>(pack->body())) return llvm::ConstantAggregateZero::get(llvm_type);

        llvm::Value* llvm_agg = llvm::UndefValue::get(llvm_type);
        if (pack->body()->isa<Bot>()) return llvm_agg;

        auto elem = lookup(pack->body());
        for (size_t i = 0, e = as_lit<u64>(pack->arity()); i != e; ++i)
            llvm_agg = irbuilder_.CreateInsertValue(llvm_agg, elem, { unsigned(i) });

        return llvm_agg;
    }

    if (def->isa<Extract>() || def->isa<Insert>()) {
        auto llvm_agg = lookup(def->op(0));
        auto llvm_idx = lookup(def->op(1));
        auto copy_to_alloca = [&] () {
            WDEF(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto alloca = emit_alloca(llvm_agg->getType(), def->name());
            irbuilder_.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { irbuilder_.getInt64(0), llvm_idx };
            auto gep = irbuilder_.CreateInBoundsGEP(alloca, args);
            return std::make_pair(alloca, gep);
        };
        auto copy_to_alloca_or_global = [&] () -> llvm::Value* {
            if (auto constant = llvm::dyn_cast<llvm::Constant>(llvm_agg)) {
                auto global = llvm::cast<llvm::GlobalVariable>(module_->getOrInsertGlobal(def->op(0)->unique_name().c_str(), llvm_agg->getType()));
                global->setInitializer(constant);
                return irbuilder_.CreateInBoundsGEP(global, { irbuilder_.getInt64(0), llvm_idx });
            }
            return copy_to_alloca().second;
        };

        if (auto extract = def->isa<Extract>()) {
            // Assemblys with more than two outputs are MemOps and have tuple type
            // and thus need their own rule here because the standard MemOp rule does not work
            //if (auto assembly = extract->agg()->isa<Assembly>()) {
                //if (assembly->type()->num_ops() > 2 && as_lit<u64>(extract->index()) != 0)
                    //return irbuilder_.CreateExtractValue(llvm_agg, {as_lit<u32>(extract->index()) - 1});
            //}

            if (is_memop(extract->agg()))
                return lookup(extract->agg());

            if (extract->agg()->type()->isa<Variadic>())
                return irbuilder_.CreateLoad(copy_to_alloca_or_global());

            // tuple/struct
            return irbuilder_.CreateExtractValue(llvm_agg, {as_lit<u32>(extract->index())});
        }

        auto insert = def->as<Insert>();
        auto val = lookup(insert->val());

        if (insert->agg()->type()->isa<Variadic>()) {
            auto p = copy_to_alloca();
            irbuilder_.CreateStore(lookup(insert->val()), p.second);
            return irbuilder_.CreateLoad(p.first);
        }
        // tuple/struct
        return irbuilder_.CreateInsertValue(llvm_agg, val, {as_lit<u32>(insert->index())});
    }

    if (auto variant = def->isa<Variant>()) {
        auto bits = compute_variant_bits(variant->type());
        auto value = lookup(variant->op(0));
        if (bits != 0) {
            auto value_bits = compute_variant_op_bits(variant->op(0)->type());
            auto bitcast = irbuilder_.CreateBitOrPointerCast(value, irbuilder_.getIntNTy(value_bits));
            return irbuilder_.CreateZExt(bitcast, irbuilder_.getIntNTy(bits));
        } else {
            auto ptr_type = llvm::PointerType::get(value->getType(), 0);
            return create_tmp_alloca(convert(variant->type()), [&] (auto alloca) {
                auto casted_ptr = irbuilder_.CreateBitCast(alloca, ptr_type);
                irbuilder_.CreateStore(value, casted_ptr);
                return irbuilder_.CreateLoad(alloca);
            });
        }
    }

    if (auto lit = def->isa<Lit>()) {
        llvm::Type* llvm_type = convert(lit->type());
        if (is_arity(lit->type())) return irbuilder_.getInt64(lit->get());

        if (lit->type()->isa<Bool>()) return irbuilder_.getInt1(lit->get<bool>());

        if (auto sint = lit->type()->isa<Sint>()) {
            switch (sint->lit_num_bits()) {
                case  8: return irbuilder_. getInt8(lit->get< s8>());
                case 16: return irbuilder_.getInt16(lit->get<s16>());
                case 32: return irbuilder_.getInt32(lit->get<s32>());
                case 64: return irbuilder_.getInt64(lit->get<s64>());
                default: THORIN_UNREACHABLE;
            }
        }

        if (auto uint = lit->type()->isa<Uint>()) {
            switch (uint->lit_num_bits()) {
                case  8: return irbuilder_. getInt8(lit->get< u8>());
                case 16: return irbuilder_.getInt16(lit->get<u16>());
                case 32: return irbuilder_.getInt32(lit->get<u32>());
                case 64: return irbuilder_.getInt64(lit->get<u64>());
                default: THORIN_UNREACHABLE;
            }
        }

        if (auto real = lit->type()->isa<Real>()) {
            switch (real->lit_num_bits()) {
                case 16: return llvm::ConstantFP::get(llvm_type, lit->get<f16>());
                case 32: return llvm::ConstantFP::get(llvm_type, lit->get<f32>());
                case 64: return llvm::ConstantFP::get(llvm_type, lit->get<f64>());
                default: THORIN_UNREACHABLE;
            }
        }

        THORIN_UNREACHABLE;
    }

    if (def->isa<Bot>())                      return llvm::UndefValue::get(convert(def->type()));
    if (auto alloc = def->isa<Alloc>())       return emit_alloc(alloc->alloced_type());
    if (auto load = def->isa<Load>())         return emit_load(load);
    if (auto store = def->isa<Store>())       return emit_store(store);
    if (auto lea = def->isa<LEA>())           return emit_lea(lea);
    //if (auto assembly = def->isa<Assembly>()) return emit_assembly(assembly);

    if (auto slot = def->isa<Slot>())
        return emit_alloca(convert(slot->alloced_type()), slot->unique_name());

    if (auto global = def->isa<Global>())
        return emit_global(global);

    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_global(const Global* global) {
    llvm::Value* val;
    if (auto lam = global->init()->isa_nominal<Lam>())
        val = fcts_[lam];
    else {
        auto llvm_type = convert(global->alloced_type());
        auto var = llvm::cast<llvm::GlobalVariable>(module_->getOrInsertGlobal(global->unique_name().c_str(), llvm_type));
        if (global->init()->isa<Bot>())
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
    if (lea->ptr_pointee()->isa<Sigma>())
        return irbuilder_.CreateStructGEP(convert(lea->ptr_pointee()), lookup(lea->ptr()), as_lit<u64>(lea->index()));

    assert(lea->ptr_pointee()->isa<Variadic>());
    llvm::Value* args[2] = { irbuilder_.getInt64(0), lookup(lea->index()) };
    return irbuilder_.CreateInBoundsGEP(lookup(lea->ptr()), args);
}

/*
llvm::Value* CodeGen::emit_assembly(const Assembly* assembly) {
    llvm::Type* res_type;

    if (auto sigma = assembly->type()->isa<Sigma>()) {
        if (sigma->num_ops() == 2)
            res_type = convert(sigma->op(1));
        else {
            auto ops = sigma->ops().skip_front();
            // don't just convert(sigma(ops)) because Thorin might normalize this to a variadic
            res_type = llvm::StructType::get(context_, llvm_ref(Array<llvm::Type*>(ops.size(), [&](auto i) { return convert(ops[i]); })));
        }
    } else {
        res_type = llvm::Type::getVoidTy(context_);
    }

    size_t num_inputs = assembly->num_inputs();
    auto input_values = Array<llvm::Value*>(num_inputs);
    auto input_types = Array<llvm::Type*>(num_inputs);
    for (size_t i = 0; i != num_inputs; ++i) {
        input_values[i] = lookup(assembly->input(i));
        input_types[i] = convert(assembly->input(i)->type());
    }

    auto fn_type = llvm::FunctionType::get(res_type, llvm_ref(input_types), false);

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
        EDEF(assembly, "constraints and input and output types of inline assembly do not match");

    auto asm_expr = llvm::InlineAsm::get(fn_type, assembly->asm_template(), constraints,
            assembly->has_sideeffects(), assembly->is_alignstack(),
            assembly->is_inteldialect() ? llvm::InlineAsm::AsmDialect::AD_Intel : llvm::InlineAsm::AsmDialect::AD_ATT);
    return irbuilder_.CreateCall(asm_expr, llvm_ref(input_values));
}
*/

unsigned CodeGen::convert_addr_space(u64 addr_space) {
    switch (addr_space) {
        case AddrSpace::Generic:  return 0;
        case AddrSpace::Global:   return 1;
        case AddrSpace::Texture:  return 2;
        case AddrSpace::Shared:   return 3;
        case AddrSpace::Constant: return 4;
        default:                  THORIN_UNREACHABLE;
    }
}

unsigned CodeGen::compute_variant_bits(const VariantType* variant) {
    unsigned total_bits = 0;
    for (auto op : variant->ops()) {
        auto type_bits = compute_variant_op_bits(op);
        if (type_bits == 0) return 0;
        total_bits = std::max(total_bits, type_bits);
    }
    return total_bits;
}

unsigned CodeGen::compute_variant_op_bits(const Def* type) {
    auto llvm_type = convert(type);
    auto layout = module_->getDataLayout();
    if (llvm_type->isPointerTy()       ||
        llvm_type->isFloatingPointTy() ||
        llvm_type->isIntegerTy())
        return layout.getTypeSizeInBits(llvm_type);
    return 0;
}

llvm::Type* CodeGen::convert(const Def* type) {
    if (auto llvm_type = types_.lookup(type)) return *llvm_type;

    assert(!type->isa<Mem>());

    if (is_arity(type))
        return types_[type] = irbuilder_.getInt64Ty();

    if (type->isa<Bool>()) return types_[type] = irbuilder_. getInt1Ty();

    if (auto sint = type->isa<Sint>()) {
        switch (sint->lit_num_bits()) {
            case 8:  return types_[type] = irbuilder_. getInt8Ty();
            case 16: return types_[type] = irbuilder_.getInt16Ty();
            case 32: return types_[type] = irbuilder_.getInt32Ty();
            case 64: return types_[type] = irbuilder_.getInt64Ty();
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto uint = type->isa<Uint>()) {
        switch (uint->lit_num_bits()) {
            case 8:  return types_[type] = irbuilder_. getInt8Ty();
            case 16: return types_[type] = irbuilder_.getInt16Ty();
            case 32: return types_[type] = irbuilder_.getInt32Ty();
            case 64: return types_[type] = irbuilder_.getInt64Ty();
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto real = type->isa<Real>()) {
        switch (real->lit_num_bits()) {
            case 16: return types_[type] = irbuilder_.getHalfTy();
            case 32: return types_[type] = irbuilder_.getFloatTy();
            case 64: return types_[type] = irbuilder_.getDoubleTy();
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto ptr = type->isa<Ptr>()) {
        auto llvm_type = llvm::PointerType::get(convert(ptr->pointee()), convert_addr_space(ptr->lit_addr_space()));
        return types_[type] = llvm_type;
    }

    if (auto variadic = type->isa<Variadic>()) {
        auto elem_type = convert(variadic->body());
        if (auto arity = isa_lit<u64>(variadic->arity()))
            return types_[type] = llvm::ArrayType::get(elem_type, *arity);
        else
            return types_[type] = llvm::ArrayType::get(elem_type, 0);
    }

    if (auto cn = type->isa<Pi>()) {
        // extract "return" type, collect all other types
        assert(cn->is_cn());
        llvm::Type* ret = nullptr;
        std::vector<llvm::Type*> domains;
        for (auto domain : cn->domains()) {
            if (domain->isa<Mem>() || domain == world().sigma()) continue;
            if (auto cn = domain->isa<Pi>()) {
                assert(cn->is_cn());
                assert(!ret && "only one 'return' supported");
                std::vector<llvm::Type*> ret_types;
                for (auto cn_domain : cn->domains()) {
                    if (cn_domain->isa<Mem>() || cn_domain == world().sigma()) continue;
                    ret_types.push_back(convert(cn_domain));
                }
                if (ret_types.size() == 0)      ret = llvm::Type::getVoidTy(context_);
                else if (ret_types.size() == 1) ret = ret_types.back();
                else                            ret = llvm::StructType::get(context_, ret_types);
            } else
                domains.push_back(convert(domain));
        }
        assert(ret);

        auto llvm_type = llvm::FunctionType::get(ret, domains, false);
        return types_[type] = llvm_type;
    }

    if (auto sigma = type->isa<Sigma>()) {
        llvm::StructType* llvm_struct = nullptr;
        if (sigma->isa_nominal()) {
            llvm_struct = llvm::StructType::create(context_);
            assert(!types_.contains(sigma) && "type already converted");
            types_[sigma] = llvm_struct;
        }

        Array<llvm::Type*> llvm_types(sigma->num_ops(), [&](auto i) { return convert(sigma->op(i)); });

        if (llvm_struct)
            llvm_struct->setBody(llvm_ref(llvm_types));
        else
            llvm_struct = llvm::StructType::get(context_, llvm_ref(llvm_types));

        return llvm_struct;
    }

    if (auto variant_type = type->isa<VariantType>()) {
        auto bits = compute_variant_bits(variant_type);
        if (bits != 0) {
            return types_[type] = irbuilder_.getIntNTy(bits);
        } else {
            auto layout = module_->getDataLayout();
            uint64_t max_size = 0;
            for (auto op : type->ops())
                max_size = std::max(max_size, layout.getTypeAllocSize(convert(op)));
            return types_[type] = llvm::ArrayType::get(irbuilder_.getInt8Ty(), max_size);
        }
    }

    THORIN_UNREACHABLE;
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

llvm::Value* CodeGen::create_tmp_alloca(llvm::Type* type, std::function<llvm::Value* (llvm::AllocaInst*)> fun) {
    // emit the alloca in the entry block
    auto alloca = emit_alloca(type, "tmp_alloca");

    // mark the lifetime of the alloca
    auto lifetime_start = llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::lifetime_start);
    auto lifetime_end   = llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::lifetime_end);
    auto addr_space = alloca->getType()->getPointerAddressSpace();
    auto void_cast = irbuilder_.CreateBitCast(alloca, llvm::PointerType::get(irbuilder_.getInt8Ty(), addr_space));

    auto layout = llvm::DataLayout(module_->getDataLayout());
    auto size = irbuilder_.getInt64(layout.getTypeAllocSize(type));

    irbuilder_.CreateCall(lifetime_start, { size, void_cast });
    auto result = fun(alloca);
    irbuilder_.CreateCall(lifetime_end, { size, void_cast });
    return result;
}

//------------------------------------------------------------------------------

#if 0
static void get_kernel_configs(Rewriter& rewriter,
    const std::vector<Lam*>& kernels,
    Cont2Config& kernel_config,
    std::function<std::unique_ptr<KernelConfig> (Lam*, Lam*)> use_callback)
{
    optimize_old(rewriter.world());

    auto externals = rewriter.world().externals();
    for (auto lam : kernels) {
        // recover the imported lam (lost after the call to opt)
        Lam* imported = nullptr;
        for (auto external : externals) {
            if (auto ex_lam = external->isa<Lam>()) {
                if (ex_lam->name() == lam->name())
                    imported = ex_lam;
            }
        }
        if (!imported) continue;

        visit_uses(lam, [&] (Lam* use) {
            auto config = use_callback(use, imported);
            if (config) {
                auto p = kernel_config.emplace(imported, std::move(config));
                assert_unused(p.second && "single kernel config entry expected");
            }
            return false;
        }, true);

        lam->destroy();
    }
}

static const Lam* get_alloc_call(const Def* def) {
    // look through casts
    while (auto conv_op = def->isa<ConvOp>())
        def = conv_op->op(0);

    auto param = def->isa<Param>();
    if (!param) return nullptr;

    auto ret = param->lam();
    if (ret->num_uses() != 1) return nullptr;

    auto use = *(ret->uses().begin());
    auto call = use.def()->isa_nominal<Lam>();
    if (!call || use.index() == 0) return nullptr;

    auto callee = call->app()->callee();
    if (callee->name() != "anydsl_alloc") return nullptr;

    return call;
}

static uint64_t get_alloc_size(const Def* def) {
    auto call = get_alloc_call(def);
    if (!call) return 0;

    // signature: anydsl_alloc(mem, i32, i64, fn(mem, &[i8]))
    auto size = call->app()->arg(2)->isa<Lit>();
    return size ? static_cast<uint64_t>(size->box().get_qu64()) : 0_u64;
}
#endif

Backends::Backends(World& world)
    : rewriters({world, world, world, world, world, world})
{
    // TODO rewrite as loop
#if 0
    // determine different parts of the world which need to be compiled differently
    Scope::for_each(world, [&] (const Scope& scope) {
        auto lam = scope.entry();
        Lam* imported = nullptr;

        if (is_passed_to_intrinsic(lam, Intrinsic::CUDA))
            imported = cuda.import(lam)->as_nominal<Lam>();
        else if (is_passed_to_intrinsic(lam, Intrinsic::NVVM))
            imported = nvvm.import(lam)->as_nominal<Lam>();
        else if (is_passed_to_intrinsic(lam, Intrinsic::OpenCL))
            imported = opencl.import(lam)->as_nominal<Lam>();
        else if (is_passed_to_intrinsic(lam, Intrinsic::AMDGPU))
            imported = amdgpu.import(lam)->as_nominal<Lam>();
        else if (is_passed_to_intrinsic(lam, Intrinsic::HLS))
            imported = hls.import(lam)->as_nominal<Lam>();
        else
            return;

        // TODO
        //imported->debug().set(lam->unique_name());
        imported->make_external();
        // TODO
        //lam->debug().set(lam->unique_name());

        //for (size_t i = 0, e = lam->num_params(); i != e; ++i)
            //imported->param(i)->debug().set(lam->param(i)->unique_name());

        kernels.emplace_back(lam);
    });

    // get the GPU kernel configurations
    if (!cuda.world().empty()   ||
        !nvvm.world().empty()   ||
        !opencl.world().empty() ||
        !amdgpu.world().empty()) {
        auto get_gpu_config = [&] (Lam* use, Lam* /* imported */) {
            // determine whether or not this kernel uses restrict pointers
            bool has_restrict = true;
            DefSet allocs;
            for (size_t i = LaunchArgs::Num, e = use->app()->num_args(); has_restrict && i != e; ++i) {
                auto arg = use->app()->arg(i);
                if (!arg->type()->isa<Ptr>()) continue;
                auto alloc = get_alloc_call(arg);
                if (!alloc) has_restrict = false;
                auto p = allocs.insert(alloc);
                has_restrict &= p.second;
            }

            auto it_config = use->app()->arg(LaunchArgs::Config)->as<Tuple>();
            if (it_config->op(0)->isa<Lit>() &&
                it_config->op(1)->isa<Lit>() &&
                it_config->op(2)->isa<Lit>()) {
                return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int> {
                    as_lit<qu32>(it_config->op(0)), as_lit<qu32>(it_config->op(1)), as_lit<qu32>(it_config->op(2)),
                }, has_restrict);
            }
            return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int> { -1, -1, -1 }, has_restrict);
        };
        get_kernel_configs(cuda,   kernels, kernel_config, get_gpu_config);
        get_kernel_configs(nvvm,   kernels, kernel_config, get_gpu_config);
        get_kernel_configs(opencl, kernels, kernel_config, get_gpu_config);
        get_kernel_configs(amdgpu, kernels, kernel_config, get_gpu_config);
    }

    // get the HLS kernel configurations
    if (!hls.world().empty()) {
        auto get_hls_config = [&] (Lam* use, Lam* imported) {
            HLSKernelConfig::Param2Size param_sizes;
            for (size_t i = 3, e = use->app()->num_args(); i != e; ++i) {
                auto arg = use->app()->arg(i);
                auto ptr_type = arg->type()->isa<Ptr>();
                if (!ptr_type) continue;
                auto size = get_alloc_size(arg);
                if (size == 0)
                    EDEF(arg, "array size is not known at compile time");
                auto elem_type = ptr_type->pointee();
                size_t multiplier = 1;
                if (!elem_type->isa<PrimType>()) {
                    if (auto variadic = elem_type->isa<Variadic>())
                        elem_type = variadic->body();
                }
                if (!elem_type->isa<PrimType>()) {
                    if (auto variadic = elem_type->isa<Variadic>(); variadic && variadic->arity()->isa<Lit>()) {
                        elem_type = variadic->body();
                        multiplier = as_lit<u64>(variadic->arity());
                    }
                }
                auto prim_type = elem_type->isa<PrimType>();
                if (!prim_type)
                    EDEF(arg, "only pointers to arrays of primitive types are supported");
                auto num_elems = size / (multiplier * num_bits(prim_type->primtype_tag()) / 8);
                // imported has type: fn (mem, fn (mem), ...)
                param_sizes.emplace(imported->param(i - 3 + 2), num_elems);
            }
            return std::make_unique<HLSKernelConfig>(param_sizes);
        };
        get_kernel_configs(hls, kernels, kernel_config, get_hls_config);
    }

    cleanup_world(world);
    codegen_prepare(world);

#endif
    codegens[CPU] = std::make_unique<CPUCodeGen>(world);

#if 0
    if (!cuda.  world().empty()) codegens[CUDA  ] = std::make_unique<CUDACodeGen  >(cuda  .world(), kernel_config);
    if (!nvvm.  world().empty()) codegens[NVVM  ] = std::make_unique<NVVMCodeGen  >(nvvm  .world(), kernel_config);
    if (!opencl.world().empty()) codegens[OpenCL] = std::make_unique<OpenCLCodeGen>(opencl.world(), kernel_config);
    if (!amdgpu.world().empty()) codegens[AMDGPU] = std::make_unique<AMDGPUCodeGen>(amdgpu.world(), kernel_config);
    if (!hls.   world().empty()) codegens[HLS   ] = std::make_unique<HLSCodeGen   >(hls   .world(), kernel_config);
#endif
}

//------------------------------------------------------------------------------

}
