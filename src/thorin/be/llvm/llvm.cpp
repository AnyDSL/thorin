#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <stdexcept>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO/Inliner.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar/ADCE.h>

#include "thorin/config.h"
#if THORIN_ENABLE_RV
#include <rv/rv.h>
#endif

#include "thorin/def.h"
#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/hls.h"
#include "thorin/be/llvm/opencl.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/util/array.h"

namespace thorin {

CodeGen::CodeGen(World& world,
                 llvm::CallingConv::ID function_calling_convention,
                 llvm::CallingConv::ID device_calling_convention,
                 llvm::CallingConv::ID kernel_calling_convention,
                 int opt, bool debug)
    : world_(world)
    , context_(new llvm::LLVMContext())
    , module_(new llvm::Module(world.name(), *context_))
    , opt_(opt)
    , debug_(debug)
    , dibuilder_(module())
    , function_calling_convention_(function_calling_convention)
    , device_calling_convention_(device_calling_convention)
    , kernel_calling_convention_(kernel_calling_convention)
    , runtime_(new Runtime(context(), module()))
{}

void CodeGen::optimize() {
    // TODO why is here a special case for opt() == 0?
    if (opt() != 0) {
        llvm::PassBuilder PB;
        llvm::PassBuilder::OptimizationLevel opt_level;

        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        switch (opt()) {
            case 0:  opt_level = llvm::PassBuilder::OptimizationLevel::O0; break;
            case 1:  opt_level = llvm::PassBuilder::OptimizationLevel::O1; break;
            case 2:  opt_level = llvm::PassBuilder::OptimizationLevel::O2; break;
            case 3:  opt_level = llvm::PassBuilder::OptimizationLevel::O3; break;
            default: opt_level = llvm::PassBuilder::OptimizationLevel::Os; break;
        }

        if (opt() == 3) {
            llvm::ModulePassManager module_pass_manager;

            //module_pass_manager.addPass(llvm::ModuleInlinerWrapperPass()); //Not compatible with LLVM v10
            llvm::CGSCCPassManager MainCGPipeline;
            MainCGPipeline.addPass(llvm::InlinerPass());
            module_pass_manager.addPass(createModuleToPostOrderCGSCCPassAdaptor(
                  createDevirtSCCRepeatedPass(
                    std::move(MainCGPipeline), 4)));

            llvm::FunctionPassManager function_pass_manager;
            function_pass_manager.addPass(llvm::ADCEPass());
            module_pass_manager.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(function_pass_manager)));

            module_pass_manager.run(module(), MAM);
        }

        llvm::ModulePassManager builder_passes = PB.buildModuleOptimizationPipeline(opt_level);
        builder_passes.run(module(), MAM);
    }
}

void CodeGen::verify() const {
#if THORIN_ENABLE_CHECKS
    if (llvm::verifyModule(module(), &llvm::errs())) {
        module().print(llvm::errs(), nullptr, false, true);
        llvm::errs() << "Broken module:\n";
        abort();
    }
#endif
}

/*
 * convert thorin Type -> llvm Type
 */

llvm::Type* CodeGen::convert(const Type* type) {
    if (auto llvm_type = types_.lookup(type)) return *llvm_type;

    assert(!type->isa<MemType>());
    llvm::Type* llvm_type;
    switch (type->tag()) {
        case PrimType_bool:                                                             llvm_type = llvm::Type::getInt1Ty  (context()); break;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  llvm_type = llvm::Type::getInt8Ty  (context()); break;
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: llvm_type = llvm::Type::getInt16Ty (context()); break;
        case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: llvm_type = llvm::Type::getInt32Ty (context()); break;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: llvm_type = llvm::Type::getInt64Ty (context()); break;
        case PrimType_pf16: case PrimType_qf16:                                         llvm_type = llvm::Type::getHalfTy  (context()); break;
        case PrimType_pf32: case PrimType_qf32:                                         llvm_type = llvm::Type::getFloatTy (context()); break;
        case PrimType_pf64: case PrimType_qf64:                                         llvm_type = llvm::Type::getDoubleTy(context()); break;
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

        case Node_ClosureType:
        case Node_FnType: {
            // extract "return" type, collect all other types
            auto fn = type->as<FnType>();
            llvm::Type* ret = nullptr;
            std::vector<llvm::Type*> ops;
            for (auto op : fn->ops()) {
                if (op->isa<MemType>() || op == world().unit()) continue;
                auto fn = op->isa<FnType>();
                if (fn && !op->isa<ClosureType>()) {
                    assert(!ret && "only one 'return' supported");
                    std::vector<llvm::Type*> ret_types;
                    for (auto fn_op : fn->ops()) {
                        if (fn_op->isa<MemType>() || fn_op == world().unit()) continue;
                        ret_types.push_back(convert(fn_op));
                    }
                    if (ret_types.size() == 0)      ret = llvm::Type::getVoidTy(context());
                    else if (ret_types.size() == 1) ret = ret_types.back();
                    else                            ret = llvm::StructType::get(context(), ret_types);
                } else
                    ops.push_back(convert(op));
            }
            assert(ret);

            if (type->tag() == Node_FnType) {
                auto llvm_type = llvm::FunctionType::get(ret, ops, false);
                return types_[type] = llvm_type;
            }

            auto env_type = convert(Closure::environment_type(world_));
            ops.push_back(env_type);
            auto fn_type = llvm::FunctionType::get(ret, ops, false);
            auto ptr_type = llvm::PointerType::get(fn_type, 0);
            llvm_type = llvm::StructType::get(context(), { ptr_type, env_type });
            return types_[type] = llvm_type;
        }

        case Node_StructType: {
            auto struct_type = type->as<StructType>();
            auto llvm_struct = llvm::StructType::create(context());

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
            llvm_type = llvm::StructType::get(context(), llvm_ref(llvm_types));
            return types_[tuple] = llvm_type;
        }

        case Node_VariantType: {
            assert(type->num_ops() > 0);
            // Max alignment/size constraints respectively in the variant type alternatives dictate the ones to use for the overall type
            size_t max_align = 0, max_size = 0;

            auto layout = module().getDataLayout();
            llvm::Type* max_align_type;
            for (auto op : type->ops()) {
                auto op_type = convert(op);
                size_t size  = layout.getTypeAllocSize(op_type);
                size_t align = layout.getABITypeAlignment(op_type);
                // Favor types that are not empty
                if (align > max_align || (align == max_align && max_align_type->isEmptyTy())) {
                    max_align_type = op_type;
                    max_align = align;
                }
                max_size = std::max(max_size, size);
            }

            auto rem_size = max_size - layout.getTypeAllocSize(max_align_type);
            auto union_type = rem_size > 0
                    ? llvm::StructType::get(context(), llvm::ArrayRef<llvm::Type*> { max_align_type, llvm::ArrayType::get(llvm::Type::getInt8Ty(context()), rem_size)})
                    : llvm::StructType::get(context(), llvm::ArrayRef<llvm::Type*> { max_align_type });

            auto tag_type = type->num_ops() < (1_u64 <<  8) ? llvm::Type::getInt8Ty (context()) :
                            type->num_ops() < (1_u64 << 16) ? llvm::Type::getInt16Ty(context()) :
                            type->num_ops() < (1_u64 << 32) ? llvm::Type::getInt32Ty(context()) :
                                                              llvm::Type::getInt64Ty(context());

            return llvm::StructType::get(context(), { union_type, tag_type });
        }

        default:
            THORIN_UNREACHABLE;
    }

    if (vector_length(type) == 1)
        return types_[type] = llvm_type;

    llvm_type = llvm::VectorType::get(llvm_type, vector_length(type));
    return types_[type] = llvm_type;
}

llvm::FunctionType* CodeGen::convert_fn_type(Continuation* continuation) {
    return llvm::cast<llvm::FunctionType>(convert(continuation->type()));
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

/*
 * emit
 */

void CodeGen::emit(std::ostream& stream) {
    llvm::raw_os_ostream llvm_stream(stream);
    emit()->print(llvm_stream, nullptr);
}

std::unique_ptr<llvm::Module>& CodeGen::emit() {
    if (debug()) {
        module().addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
        // Darwin only supports dwarf2
        if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin())
            module().addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
        dicompile_unit_ = dibuilder_.createCompileUnit(llvm::dwarf::DW_LANG_C, dibuilder_.createFile(world_.name(), llvm::StringRef()), "Impala", opt() > 0, llvm::StringRef(), 0);
    }

    Scope::for_each(world_, [&] (const Scope& scope) { emit(scope); });

    if (debug()) dibuilder_.finalize();

#if THORIN_ENABLE_RV
    for (auto [width, fct, call] : vec_todo_)
        emit_vectorize(width, fct, call);
    vec_todo_.clear();

    rv::lowerIntrinsics(module());
#endif

#if THORIN_ENABLE_CHECKS
    llvm::verifyModule(module());
#endif
    optimize();

    return module_;
}

llvm::Function* CodeGen::emit_function_decl(Continuation* continuation) {
    std::string name = (continuation->is_exported() || continuation->empty()) ? continuation->name() : continuation->unique_name();
    auto f = llvm::cast<llvm::Function>(module().getOrInsertFunction(name, convert_fn_type(continuation)).getCallee()->stripPointerCasts());

#ifdef _MSC_VER
    // set dll storage class for MSVC
    if (!entry_ && llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
        if (continuation->empty()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        } else if (continuation->is_exported()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }
#endif

    // set linkage
    if (continuation->empty() || continuation->is_exported())
        f->setLinkage(llvm::Function::ExternalLinkage);
    else
        f->setLinkage(llvm::Function::InternalLinkage);

    // set calling convention
    if (continuation->is_exported()) {
        f->setCallingConv(kernel_calling_convention_);
        emit_function_decl_hook(continuation, f);
    } else {
        if (continuation->cc() == CC::Device)
            f->setCallingConv(device_calling_convention_);
        else
            f->setCallingConv(function_calling_convention_);
    }

    return f;
}

void CodeGen::emit(const Scope& scope) {
    entry_ = scope.entry();
    assert(entry_->is_returning());
    auto fct = llvm::cast<llvm::Function>(emit(entry_));

    llvm::DIScope* discope = dicompile_unit_;
    if (debug()) {
        auto src_file = llvm::sys::path::filename(entry_->loc().file);
        auto src_dir = llvm::sys::path::parent_path(entry_->loc().file);
        auto difile = dibuilder_.createFile(src_file, src_dir);
        auto disub_program = dibuilder_.createFunction(
            discope, fct->getName(), fct->getName(), difile, entry_->loc().begin.row,
            dibuilder_.createSubroutineType(dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
            entry_->loc().begin.row,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition | (opt() > 0 ? llvm::DISubprogram::SPFlagOptimized : llvm::DISubprogram::SPFlagZero));
        fct->setSubprogram(disub_program);
        discope = disub_program;
    }

    cont2llvm_.clear();
    auto conts = schedule(scope);

    // map all bb-like continuations to llvm bb stubs and handle params/phis
    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;

        auto bb = llvm::BasicBlock::Create(context(), cont->name().c_str(), fct);
        auto [i, succ] = cont2llvm_.emplace(cont, std::pair(bb, std::make_unique<llvm::IRBuilder<>>(context())));
        assert(succ);
        auto& irbuilder = *i->second.second;
        irbuilder.SetInsertPoint(bb);
        if (debug())
            irbuilder.SetCurrentDebugLocation(llvm::DebugLoc::get(cont->loc().begin.row, cont->loc().begin.row, discope));

        if (entry_ == cont) {
            auto arg = fct->arg_begin();
            for (auto param : entry_->params()) {
                if (is_mem(param) || is_unit(param)) {
                    def2llvm_[param] = nullptr;
                } else if (param->order() == 0) {
                    auto argv = &*arg;
                    auto value = map_param(fct, argv, param);
                    if (value == argv) {
                        arg->setName(param->unique_name()); // use param
                        def2llvm_[param] = &*arg++;
                    } else {
                        def2llvm_[param] = value;           // use provided value
                    }
                }
            }
        } else {
            for (auto param : cont->params()) {
                if (is_mem(param) || is_unit(param)) {
                    def2llvm_[param] = nullptr;
                } else {
                    // do not bother reserving anything (the 0 below) - it's a tiny optimization nobody cares about
                    auto phi = irbuilder.CreatePHI(convert(param->type()), 0, param->name().c_str());
                    def2llvm_[param] = phi;
                }
            }
        }
    }

    Scheduler new_scheduler(scope);
    swap(scheduler_, new_scheduler);

    emit_function_start(entry_);

    for (auto cont : conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;
        assert(cont == entry_ || cont->is_basicblock());
        emit_epilogue(cont);
    }
}

void CodeGen::emit_epilogue(Continuation* continuation) {
    auto&& bb_ib = cont2llvm_[continuation];
    auto bb = bb_ib->first;
    auto& irbuilder = *bb_ib->second;

    if (continuation->callee() == entry_->ret_param()) { // return
        std::vector<llvm::Value*> values;
        std::vector<llvm::Type *> types;

        for (auto arg : continuation->args()) {
            if (auto val = emit_unsafe(arg)) {
                values.emplace_back(val);
                types.emplace_back(val->getType());
            }
        }

        switch (values.size()) {
            case 0:  irbuilder.CreateRetVoid();      break;
            case 1:  irbuilder.CreateRet(values[0]); break;
            default:
                llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context(), types));

                for (size_t i = 0, e = values.size(); i != e; ++i)
                    agg = irbuilder.CreateInsertValue(agg, values[i], { unsigned(i) });

                irbuilder.CreateRet(agg);
        }
    } else if (continuation->callee() == world().branch()) {
        auto cond = emit(continuation->arg(0));
        auto tbb = cont2bb(continuation->arg(1)->as_continuation());
        auto fbb = cont2bb(continuation->arg(2)->as_continuation());
        irbuilder.CreateCondBr(cond, tbb, fbb);
    } else if (continuation->callee()->isa<Continuation>() &&
                continuation->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
        auto val = emit(continuation->arg(0));
        auto otherwise_bb = cont2bb(continuation->arg(1)->as_continuation());
        auto match = irbuilder.CreateSwitch(val, otherwise_bb, continuation->num_args() - 2);
        for (size_t i = 2; i < continuation->num_args(); i++) {
            auto arg = continuation->arg(i)->as<Tuple>();
            auto case_const = llvm::cast<llvm::ConstantInt>(emit(arg->op(0)));
            auto case_bb    = cont2bb(arg->op(1)->as_continuation());
            match->addCase(case_const, case_bb);
        }
    } else if (continuation->callee()->isa<Bottom>()) {
        irbuilder.CreateUnreachable();
    } else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        for (size_t i = 0, e = continuation->num_args(); i != e; ++i) {
            if (auto val = emit_unsafe(continuation->arg(i))) emit_phi_arg(irbuilder, callee->param(i), val);
        }
        irbuilder.CreateBr(cont2bb(callee));
    } else if (auto callee = continuation->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
        auto ret_continuation = emit_intrinsic(irbuilder, continuation);
        irbuilder.CreateBr(cont2bb(ret_continuation));
    } else { // function/closure call
        // put all first-order args into an array
        std::vector<llvm::Value*> args;
        const Def* ret_arg = nullptr;
        for (auto arg : continuation->args()) {
            if (arg->order() == 0) {
                if (auto val = emit_unsafe(arg))
                    args.push_back(val);
            } else {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        llvm::CallInst* call = nullptr;
        if (auto callee = continuation->callee()->isa_continuation()) {
            call = irbuilder.CreateCall(emit(callee), args);
            if (callee->is_exported())
                call->setCallingConv(kernel_calling_convention_);
            else if (callee->cc() == CC::Device)
                call->setCallingConv(device_calling_convention_);
            else
                call->setCallingConv(function_calling_convention_);
        } else {
            // must be a closure
            auto closure = emit(callee);
            args.push_back(irbuilder.CreateExtractValue(closure, 1));
            call = irbuilder.CreateCall(irbuilder.CreateExtractValue(closure, 0), args);
        }

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
            irbuilder.CreateBr(cont2bb(succ));
        } else if (n == 1) {
            irbuilder.CreateBr(cont2bb(succ));
            emit_phi_arg(irbuilder, last_param, call);
        } else {
            Array<llvm::Value*> extracts(n);
            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                extracts[j] = irbuilder.CreateExtractValue(call, unsigned(j));
                j++;
            }

            irbuilder.CreateBr(cont2bb(succ));

            for (size_t i = 0, j = 0; i != succ->num_params(); ++i) {
                auto param = succ->param(i);
                if (is_mem(param) || is_unit(param))
                    continue;
                emit_phi_arg(irbuilder, param, extracts[j]);
                j++;
            }
        }
    }

    // new insert point is just before the terminator for all other instructions we have to add later on
    irbuilder.SetInsertPoint(bb->getTerminator());
}

llvm::Value* CodeGen::emit_unsafe(const Def* def) {
    if (auto llvm = def2llvm_.lookup(def)) return *llvm;
    if (auto cont = def->isa_continuation()) return def2llvm_[cont] = emit_function_decl(cont);

    auto llvm = emit_(def);
    return def2llvm_[def] = llvm;
}

llvm::Value* CodeGen::emit(const Def* def) {
    auto res = emit_unsafe(def);
    assert(res);
    return res;
}

llvm::Value* CodeGen::emit_(const Def* def) {
    auto place = is_const(def) ? entry_ : scheduler_.smart(def);
    auto& irbuilder = *cont2llvm_[place]->second;

    // TODO
    //if (debug())
        //irbuilder.SetCurrentDebugLocation(llvm::DebugLoc::get(def->loc().begin.row, def->loc().begin.row, discope));

    if (auto bin = def->isa<BinOp>()) {
        llvm::Value* lhs = emit(bin->lhs());
        llvm::Value* rhs = emit(bin->rhs());
        const char* name = bin->name().c_str();

        if (auto cmp = bin->isa<Cmp>()) {
            auto type = cmp->lhs()->type();
            if (is_type_s(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder.CreateICmpSGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder.CreateICmpSGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder.CreateICmpSLT(lhs, rhs, name);
                    case Cmp_le: return irbuilder.CreateICmpSLE(lhs, rhs, name);
                }
            } else if (is_type_u(type) || is_type_bool(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder.CreateICmpNE (lhs, rhs, name);
                    case Cmp_gt: return irbuilder.CreateICmpUGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder.CreateICmpUGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder.CreateICmpULT(lhs, rhs, name);
                    case Cmp_le: return irbuilder.CreateICmpULE(lhs, rhs, name);
                }
            } else if (is_type_f(type)) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder.CreateFCmpOEQ(lhs, rhs, name);
                    case Cmp_ne: return irbuilder.CreateFCmpUNE(lhs, rhs, name);
                    case Cmp_gt: return irbuilder.CreateFCmpOGT(lhs, rhs, name);
                    case Cmp_ge: return irbuilder.CreateFCmpOGE(lhs, rhs, name);
                    case Cmp_lt: return irbuilder.CreateFCmpOLT(lhs, rhs, name);
                    case Cmp_le: return irbuilder.CreateFCmpOLE(lhs, rhs, name);
                }
            } else if (type->isa<PtrType>()) {
                switch (cmp->cmp_tag()) {
                    case Cmp_eq: return irbuilder.CreateICmpEQ (lhs, rhs, name);
                    case Cmp_ne: return irbuilder.CreateICmpNE (lhs, rhs, name);
                    default: THORIN_UNREACHABLE;
                }
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            auto type = arithop->type();
            bool q = is_type_q(arithop->type()); // quick? -> nsw/nuw/fast float

            if (is_type_f(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder.CreateFAdd(lhs, rhs, name);
                    case ArithOp_sub: return irbuilder.CreateFSub(lhs, rhs, name);
                    case ArithOp_mul: return irbuilder.CreateFMul(lhs, rhs, name);
                    case ArithOp_div: return irbuilder.CreateFDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder.CreateFRem(lhs, rhs, name);
                    case ArithOp_and:
                    case ArithOp_or:
                    case ArithOp_xor:
                    case ArithOp_shl:
                    case ArithOp_shr: THORIN_UNREACHABLE;
                }
            }

            if (is_type_s(type) || is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder.CreateAdd (lhs, rhs, name, false, q);
                    case ArithOp_sub: return irbuilder.CreateSub (lhs, rhs, name, false, q);
                    case ArithOp_mul: return irbuilder.CreateMul (lhs, rhs, name, false, q);
                    case ArithOp_div: return irbuilder.CreateSDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder.CreateSRem(lhs, rhs, name);
                    case ArithOp_and: return irbuilder.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return irbuilder.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return irbuilder.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return irbuilder.CreateShl (lhs, rhs, name, false, q);
                    case ArithOp_shr: return irbuilder.CreateAShr(lhs, rhs, name);
                }
            }
            if (is_type_u(type) || is_type_bool(type)) {
                switch (arithop->arithop_tag()) {
                    case ArithOp_add: return irbuilder.CreateAdd (lhs, rhs, name, q, false);
                    case ArithOp_sub: return irbuilder.CreateSub (lhs, rhs, name, q, false);
                    case ArithOp_mul: return irbuilder.CreateMul (lhs, rhs, name, q, false);
                    case ArithOp_div: return irbuilder.CreateUDiv(lhs, rhs, name);
                    case ArithOp_rem: return irbuilder.CreateURem(lhs, rhs, name);
                    case ArithOp_and: return irbuilder.CreateAnd (lhs, rhs, name);
                    case ArithOp_or:  return irbuilder.CreateOr  (lhs, rhs, name);
                    case ArithOp_xor: return irbuilder.CreateXor (lhs, rhs, name);
                    case ArithOp_shl: return irbuilder.CreateShl (lhs, rhs, name, q, false);
                    case ArithOp_shr: return irbuilder.CreateLShr(lhs, rhs, name);
                }
            }
        }
    }

    if (auto conv = def->isa<ConvOp>()) {
        auto from = emit(conv->from());
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();
        auto to = convert(dst_type);

        if (conv->isa<Cast>()) {
            if (src_type->isa<PtrType>() && dst_type->isa<PtrType>()) {
                return irbuilder.CreatePointerCast(from, to);
            }
            if (src_type->isa<PtrType>()) {
                assert(is_type_i(dst_type) || is_type_bool(dst_type));
                return irbuilder.CreatePtrToInt(from, to);
            }
            if (dst_type->isa<PtrType>()) {
                assert(is_type_i(src_type) || is_type_bool(src_type));
                return irbuilder.CreateIntToPtr(from, to);
            }

            auto src = src_type->as<PrimType>();
            auto dst = dst_type->as<PrimType>();

            if (is_type_f(src) && is_type_f(dst)) {
                assert(num_bits(src->primtype_tag()) != num_bits(dst->primtype_tag()));
                return irbuilder.CreateFPCast(from, to);
            }
            if (is_type_f(src)) {
                if (is_type_s(dst))
                    return irbuilder.CreateFPToSI(from, to);
                return irbuilder.CreateFPToUI(from, to);
            }
            if (is_type_f(dst)) {
                if (is_type_s(src))
                    return irbuilder.CreateSIToFP(from, to);
                return irbuilder.CreateUIToFP(from, to);
            }

            if (num_bits(src->primtype_tag()) > num_bits(dst->primtype_tag())) {
                if (is_type_i(src) && (is_type_i(dst) || is_type_bool(dst)))
                    return irbuilder.CreateTrunc(from, to);
            } else if (num_bits(src->primtype_tag()) < num_bits(dst->primtype_tag())) {
                if ( is_type_s(src)                       && is_type_i(dst)) return irbuilder.CreateSExt(from, to);
                if ((is_type_u(src) || is_type_bool(src)) && is_type_i(dst)) return irbuilder.CreateZExt(from, to);
            } else if (is_type_i(src) && is_type_i(dst)) {
                assert(num_bits(src->primtype_tag()) == num_bits(dst->primtype_tag()));
                return from;
            }

            assert(false && "unsupported cast");
        }

        if (conv->isa<Bitcast>())
            return emit_bitcast(irbuilder, conv->from(), dst_type);
    }

    if (auto select = def->isa<Select>()) {
        if (def->type()->isa<FnType>())
            return nullptr;

        llvm::Value* cond = emit(select->cond());
        llvm::Value* tval = emit(select->tval());
        llvm::Value* fval = emit(select->fval());
        return irbuilder.CreateSelect(cond, tval, fval);
    }

    if (auto align_of = def->isa<AlignOf>()) {
        auto type = convert(align_of->of());
        return irbuilder.getInt64(module().getDataLayout().getABITypeAlignment(type));
    }

    if (auto size_of = def->isa<SizeOf>()) {
        auto type = convert(size_of->of());
        return irbuilder.getInt64(module().getDataLayout().getTypeAllocSize(type));
    }

    if (auto array = def->isa<DefiniteArray>()) {
        auto type = llvm::cast<llvm::ArrayType>(convert(array->type()));

        // Try to emit it as a constant first
        Array<llvm::Constant*> consts(array->num_ops());
        bool all_consts = true;
        for (size_t i = 0, n = consts.size(); i != n; ++i) {
            consts[i] = llvm::dyn_cast<llvm::Constant>(emit(array->op(i)));
            if (!consts[i]) {
                all_consts = false;
                break;
            }
        }
        if (all_consts)
            return llvm::ConstantArray::get(type, llvm_ref(consts));

        world().wdef(def, "slow: alloca and loads/stores needed for definite array '{}'", def);
        auto alloca = emit_alloca(irbuilder, type, array->name());

        u64 i = 0;
        llvm::Value* args[2] = { irbuilder.getInt64(0), nullptr };
        for (auto op : array->ops()) {
            args[1] = irbuilder.getInt64(i++);
            auto gep = irbuilder.CreateInBoundsGEP(alloca, args, op->name().c_str());
            irbuilder.CreateStore(emit(op), gep);
        }

        return irbuilder.CreateLoad(alloca);
    }

    if (auto array = def->isa<IndefiniteArray>())
        return llvm::UndefValue::get(convert(array->type()));

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>() || def->isa<Closure>());
        if (is_unit(agg)) return nullptr;

        llvm::Value* llvm_agg = llvm::UndefValue::get(convert(agg->type()));
        if (def->isa<Vector>()) {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder.CreateInsertElement(llvm_agg, emit(agg->op(i)), irbuilder.getInt32(i));
        } else if (auto closure = def->isa<Closure>()) {
            auto closure_fn = irbuilder.CreatePointerCast(emit(agg->op(0)), llvm_agg->getType()->getStructElementType(0));
            auto val = agg->op(1);
            llvm::Value* env = nullptr;
            if (is_thin(closure->op(1)->type())) {
                if (is_type_unit(val->type())) {
                    env = emit(world_.bottom(Closure::environment_type(world_)));
                } else {
                    env = emit(world_.cast(Closure::environment_type(world_), val));
                }
            } else {
                world().wdef(def, "closure '{}' is leaking memory, type '{}' is too large", def, agg->op(1)->type());
                auto alloc = emit_alloc(irbuilder, val->type(), nullptr);
                irbuilder.CreateStore(emit(val), alloc);
                env = irbuilder.CreatePtrToInt(alloc, convert(Closure::environment_type(world_)));
            }
            llvm_agg = irbuilder.CreateInsertValue(llvm_agg, closure_fn, 0);
            llvm_agg = irbuilder.CreateInsertValue(llvm_agg, env, 1);
        } else {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder.CreateInsertValue(llvm_agg, emit(agg->op(i)), { unsigned(i) });
        }

        return llvm_agg;
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto llvm_agg = emit_unsafe(aggop->agg());
        auto llvm_idx = emit(aggop->index());

        bool mem = false;
        if (auto tt = aggop->agg()->type()->isa<TupleType>(); tt && tt->op(0)->isa<MemType>()) mem = true;

        auto copy_to_alloca = [&] () {
            world().wdef(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto alloca = emit_alloca(irbuilder, llvm_agg->getType(), aggop->name());
            irbuilder.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { irbuilder.getInt64(0), llvm_idx };
            auto gep = irbuilder.CreateInBoundsGEP(alloca, args);
            return std::make_pair(alloca, gep);
        };
        auto copy_to_alloca_or_global = [&] () -> llvm::Value* {
            if (auto constant = llvm::dyn_cast<llvm::Constant>(llvm_agg)) {
                auto global = llvm::cast<llvm::GlobalVariable>(module().getOrInsertGlobal(aggop->agg()->unique_name().c_str(), llvm_agg->getType()));
                global->setLinkage(llvm::GlobalValue::InternalLinkage);
                global->setInitializer(constant);
                return irbuilder.CreateInBoundsGEP(global, { irbuilder.getInt64(0), llvm_idx });
            }
            return copy_to_alloca().second;
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (aggop->agg()->type()->isa<ArrayType>())
                return irbuilder.CreateLoad(copy_to_alloca_or_global());

            if (extract->agg()->type()->isa<VectorType>())
                return irbuilder.CreateExtractElement(llvm_agg, llvm_idx);

            // tuple/struct
            if (is_mem(extract)) return nullptr;

            unsigned offset = 0;
            if (mem) {
                if (aggop->agg()->type()->num_ops() == 2) return llvm_agg;
                offset = 1;
            }

            return irbuilder.CreateExtractValue(llvm_agg, {primlit_value<unsigned>(aggop->index()) - offset});
        }

        auto insert = def->as<Insert>();
        auto value = emit(insert->value());

        // TODO deal with mem - but I think for now this case shouldn't happen

        if (insert->agg()->type()->isa<ArrayType>()) {
            auto p = copy_to_alloca();
            irbuilder.CreateStore(emit(aggop->as<Insert>()->value()), p.second);
            return irbuilder.CreateLoad(p.first);
        }
        if (insert->agg()->type()->isa<VectorType>())
            return irbuilder.CreateInsertElement(llvm_agg, emit(aggop->as<Insert>()->value()), llvm_idx);
        // tuple/struct
        return irbuilder.CreateInsertValue(llvm_agg, value, {primlit_value<unsigned>(aggop->index())});
    }

    if (auto variant_index = def->isa<VariantIndex>()) {
        auto llvm_value = emit(variant_index->op(0));
        auto tag_value = irbuilder.CreateExtractValue(llvm_value, { 1 });
        return irbuilder.CreateIntCast(tag_value, convert(variant_index->type()), false);
    }
    if (auto variant_extract = def->isa<VariantExtract>()) {
        auto variant_value = variant_extract->op(0);
        auto llvm_value    = emit(variant_value);
        auto payload_value = irbuilder.CreateExtractValue(llvm_value, { 0 });

        auto target_type = convert(variant_value->type()->op(variant_extract->index()));
        return create_tmp_alloca(irbuilder, payload_value->getType(), [&] (llvm::AllocaInst* alloca) {
            irbuilder.CreateStore(payload_value, alloca);
            auto addr_space = alloca->getType()->getPointerAddressSpace();
            auto payload_addr = irbuilder.CreatePointerCast(alloca, llvm::PointerType::get(target_type, addr_space));
            return irbuilder.CreateLoad(payload_addr);
        });
    }
    if (auto variant_ctor = def->isa<Variant>()) {
        auto llvm_type = convert(variant_ctor->type());

        auto tag_value = irbuilder.getIntN(llvm_type->getStructElementType(1)->getScalarSizeInBits(), variant_ctor->index());
        auto payload_value = emit(variant_ctor->op(0));

        return create_tmp_alloca(irbuilder, llvm_type, [&] (llvm::AllocaInst* alloca) {
            auto tag_addr = irbuilder.CreateInBoundsGEP(alloca, { irbuilder.getInt32(0), irbuilder.getInt32(1) });
            irbuilder.CreateStore(tag_value, tag_addr);

            // Do not store anything if the payload is unit
            if (!is_type_unit(variant_ctor->op(0)->type())) {
                auto payload_addr = irbuilder.CreatePointerCast(
                    irbuilder.CreateInBoundsGEP(alloca, { irbuilder.getInt32(0), irbuilder.getInt32(0) }),
                    llvm::PointerType::get(payload_value->getType(), alloca->getType()->getPointerAddressSpace()));
                irbuilder.CreateStore(payload_value, payload_addr);
            }
            return irbuilder.CreateLoad(alloca);
        });
    }

    if (auto primlit = def->isa<PrimLit>()) {
        llvm::Type* llvm_type = convert(primlit->type());
        Box box = primlit->value();

        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     return irbuilder. getInt1(box.get_bool());
            case PrimType_ps8:  case PrimType_qs8:  return irbuilder. getInt8(box. get_s8());
            case PrimType_pu8:  case PrimType_qu8:  return irbuilder. getInt8(box. get_u8());
            case PrimType_ps16: case PrimType_qs16: return irbuilder.getInt16(box.get_s16());
            case PrimType_pu16: case PrimType_qu16: return irbuilder.getInt16(box.get_u16());
            case PrimType_ps32: case PrimType_qs32: return irbuilder.getInt32(box.get_s32());
            case PrimType_pu32: case PrimType_qu32: return irbuilder.getInt32(box.get_u32());
            case PrimType_ps64: case PrimType_qs64: return irbuilder.getInt64(box.get_s64());
            case PrimType_pu64: case PrimType_qu64: return irbuilder.getInt64(box.get_u64());
            case PrimType_pf16: case PrimType_qf16: return llvm::ConstantFP::get(llvm_type, box.get_f16());
            case PrimType_pf32: case PrimType_qf32: return llvm::ConstantFP::get(llvm_type, box.get_f32());
            case PrimType_pf64: case PrimType_qf64: return llvm::ConstantFP::get(llvm_type, box.get_f64());
        }
    }

    if (auto bottom = def->isa<Bottom>())
        return llvm::UndefValue::get(convert(bottom->type()));

    if (auto alloc = def->isa<Alloc>()) {
        emit_unsafe(alloc->mem());
        return emit_alloc(irbuilder, alloc->alloced_type(), alloc->extra());
    }

    if (auto load = def->isa<Load>())           return emit_load(irbuilder, load);
    if (auto store = def->isa<Store>())         return emit_store(irbuilder, store);
    if (auto lea = def->isa<LEA>())             return emit_lea(irbuilder, lea);
    if (auto assembly = def->isa<Assembly>())   return emit_assembly(irbuilder, assembly);
    if (def->isa<Enter>())                      return nullptr;

    if (auto slot = def->isa<Slot>())
        return emit_alloca(irbuilder, convert(slot->type()->as<PtrType>()->pointee()), slot->unique_name());

    if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(convert(vector->type()));
        for (size_t i = 0, e = vector->num_ops(); i != e; ++i)
            vec = irbuilder.CreateInsertElement(vec, emit(vector->op(i)), emit(world_.literal_pu32(i, vector->loc())));

        return vec;
    }

    if (auto global = def->isa<Global>())
        return emit_global(global);

    THORIN_UNREACHABLE;
}

void CodeGen::emit_phi_arg(llvm::IRBuilder<>& irbuilder, const Param* param, llvm::Value* value) {
    llvm::cast<llvm::PHINode>(*def2llvm_[param])->addIncoming(value, irbuilder.GetInsertBlock());
}

/*
 * emit: special overridable methods
 */

llvm::Value* CodeGen::emit_alloc(llvm::IRBuilder<>& irbuilder, const Type* type, const Def* extra) {
    auto llvm_malloc = runtime_->get(get_alloc_name().c_str());
    auto alloced_type = convert(type);
    llvm::CallInst* void_ptr;
    auto layout = module().getDataLayout();
    if (auto array = type->isa<IndefiniteArrayType>()) {
        assert(extra);
        auto size = irbuilder.CreateAdd(
                irbuilder.getInt64(layout.getTypeAllocSize(alloced_type)),
                irbuilder.CreateMul(irbuilder.CreateIntCast(emit(extra), irbuilder.getInt64Ty(), false),
                                     irbuilder.getInt64(layout.getTypeAllocSize(convert(array->elem_type())))));
        llvm::Value* malloc_args[] = { irbuilder.getInt32(0), size };
        void_ptr = irbuilder.CreateCall(llvm_malloc, malloc_args);
    } else {
        llvm::Value* malloc_args[] = { irbuilder.getInt32(0), irbuilder.getInt64(layout.getTypeAllocSize(alloced_type)) };
        void_ptr = irbuilder.CreateCall(llvm_malloc, malloc_args);
    }

    return irbuilder.CreatePointerCast(void_ptr, llvm::PointerType::get(alloced_type, 0));
}

llvm::AllocaInst* CodeGen::emit_alloca(llvm::IRBuilder<>& irbuilder, llvm::Type* type, const std::string& name) {
    // Emit the alloca in the entry block
    auto entry = &irbuilder.GetInsertBlock()->getParent()->getEntryBlock();
    auto layout = module().getDataLayout();
    llvm::AllocaInst* alloca;
    if (entry->empty())
        alloca = new llvm::AllocaInst(type, layout.getAllocaAddrSpace(), nullptr, name, entry);
    else
        alloca = new llvm::AllocaInst(type, layout.getAllocaAddrSpace(), nullptr, name, entry->getFirstNonPHIOrDbg());
    alloca->setAlignment(llvm::MaybeAlign(layout.getABITypeAlignment(type)));
    return alloca;
}

llvm::Value* CodeGen::emit_bitcast(llvm::IRBuilder<>& irbuilder, const Def* val, const Type* dst_type) {
    auto from = emit(val);
    auto src_type = val->type();
    auto to = convert(dst_type);
    if (from->getType()->isAggregateType() || to->isAggregateType())
        world().edef(val, "bitcast from or to aggregate types not allowed: bitcast from '{}' to '{}'", src_type, dst_type);
    if (src_type->isa<PtrType>() && dst_type->isa<PtrType>())
        return irbuilder.CreatePointerCast(from, to);
    return irbuilder.CreateBitCast(from, to);
}

llvm::Value* CodeGen::emit_global(const Global* global) {
    llvm::Value* val;
    if (auto continuation = global->init()->isa_continuation())
        val = emit(continuation);
    else {
        auto llvm_type = convert(global->alloced_type());
        auto var = llvm::cast<llvm::GlobalVariable>(module().getOrInsertGlobal(global->unique_name().c_str(), llvm_type));
        var->setConstant(!global->is_mutable());
        var->setLinkage(llvm::GlobalValue::InternalLinkage);
        if (global->init()->isa<Bottom>())
            var->setInitializer(llvm::Constant::getNullValue(llvm_type)); // HACK
        else
            var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));
        val = var;
    }
    return val;
}

llvm::GlobalVariable* CodeGen::emit_global_variable(llvm::Type* type, const std::string& name, unsigned addr_space, bool init_undef) {
    auto init = init_undef ? llvm::UndefValue::get(type) : llvm::Constant::getNullValue(type);
    return new llvm::GlobalVariable(module(), type, false, llvm::GlobalValue::InternalLinkage, init, name, nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
}

llvm::Value* CodeGen::emit_load(llvm::IRBuilder<>& irbuilder, const Load* load) {
    emit_unsafe(load->mem());
    auto ptr = emit(load->ptr());
    auto result = irbuilder.CreateLoad(ptr);
    auto align = module().getDataLayout().getABITypeAlignment(ptr->getType()->getPointerElementType());
    result->setAlignment(llvm::MaybeAlign(align));
    return result;
}

llvm::Value* CodeGen::emit_store(llvm::IRBuilder<>& irbuilder, const Store* store) {
    emit_unsafe(store->mem());
    auto ptr = emit(store->ptr());
    auto result = irbuilder.CreateStore(emit(store->val()), ptr);
    auto align = module().getDataLayout().getABITypeAlignment(ptr->getType()->getPointerElementType());
    result->setAlignment(llvm::MaybeAlign(align));
    return nullptr;
}

llvm::Value* CodeGen::emit_lea(llvm::IRBuilder<>& irbuilder, const LEA* lea) {
    if (lea->ptr_pointee()->isa<TupleType>() || lea->ptr_pointee()->isa<StructType>())
        return irbuilder.CreateStructGEP(convert(lea->ptr_pointee()), emit(lea->ptr()), primlit_value<u32>(lea->index()));

    assert(lea->ptr_pointee()->isa<ArrayType>() || lea->ptr_pointee()->isa<VectorType>());
    llvm::Value* args[2] = { irbuilder.getInt64(0), emit(lea->index()) };
    return irbuilder.CreateInBoundsGEP(emit(lea->ptr()), args);
}

llvm::Value* CodeGen::emit_assembly(llvm::IRBuilder<>& irbuilder, const Assembly* assembly) {
    emit_unsafe(assembly->mem());
    auto out_type = assembly->type();
    llvm::Type* res_type;
    bool mem_only = false;

    if (out_type->isa<TupleType>()) {
        if (out_type->num_ops() == 2)
            res_type = convert(assembly->type()->op(1));
        else
            res_type = convert(world().tuple_type(assembly->type()->ops().skip_front()));
    } else {
        res_type = llvm::Type::getVoidTy(context());
        mem_only = true;
    }

    size_t num_inputs = assembly->num_inputs();
    auto input_values = Array<llvm::Value*>(num_inputs);
    auto input_types = Array<llvm::Type*>(num_inputs);
    for (size_t i = 0; i != num_inputs; ++i) {
        input_values[i] = emit(assembly->input(i));
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
        world().edef(assembly, "constraints and input and output types of inline assembly do not match");

    auto asm_expr = llvm::InlineAsm::get(fn_type, assembly->asm_template(), constraints,
            assembly->has_sideeffects(), assembly->is_alignstack(),
            assembly->is_inteldialect() ? llvm::InlineAsm::AsmDialect::AD_Intel : llvm::InlineAsm::AsmDialect::AD_ATT);
    auto res = irbuilder.CreateCall(asm_expr, llvm_ref(input_values));

    return mem_only ? nullptr : res;
}

/*
 * emit intrinsic
 */

Continuation* CodeGen::emit_intrinsic(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    auto callee = continuation->callee()->as_continuation();
    switch (callee->intrinsic()) {
        case Intrinsic::Atomic:      return emit_atomic(irbuilder, continuation);
        case Intrinsic::AtomicLoad:  return emit_atomic_load(irbuilder, continuation);
        case Intrinsic::AtomicStore: return emit_atomic_store(irbuilder, continuation);
        case Intrinsic::CmpXchg:     return emit_cmpxchg(irbuilder, continuation);
        case Intrinsic::Reserve:     return emit_reserve(irbuilder, continuation);
        case Intrinsic::CUDA:        return runtime_->emit_host_code(*this, irbuilder, Runtime::CUDA_PLATFORM,   ".cu",     continuation);
        case Intrinsic::NVVM:        return runtime_->emit_host_code(*this, irbuilder, Runtime::CUDA_PLATFORM,   ".nvvm",   continuation);
        case Intrinsic::OpenCL:      return runtime_->emit_host_code(*this, irbuilder, Runtime::OPENCL_PLATFORM, ".cl",     continuation);
        case Intrinsic::AMDGPU:      return runtime_->emit_host_code(*this, irbuilder, Runtime::HSA_PLATFORM,    ".amdgpu", continuation);
        case Intrinsic::HLS:         return emit_hls(irbuilder, continuation);
        case Intrinsic::Parallel:    return emit_parallel(irbuilder, continuation);
        case Intrinsic::Fibers:      return emit_fibers(irbuilder, continuation);
        case Intrinsic::Spawn:       return emit_spawn(irbuilder, continuation);
        case Intrinsic::Sync:        return emit_sync(irbuilder, continuation);
#if THORIN_ENABLE_RV
        case Intrinsic::Vectorize:   return emit_vectorize_continuation(irbuilder, continuation);
#else
        case Intrinsic::Vectorize:   throw std::runtime_error("rebuild with RV support");
#endif
        default: THORIN_UNREACHABLE;
    }
}

Continuation* CodeGen::emit_atomic(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->num_args() == 7 && "required arguments are missing");
    // atomic tag: Xchg Add Sub And Nand Or Xor Max Min UMax UMin FAdd FSub
    u32 binop_tag = continuation->arg(1)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicRMWInst::BinOp::Xchg) <= int(binop_tag) && int(binop_tag) <= int(llvm::AtomicRMWInst::BinOp::FSub) && "unsupported atomic");
    auto binop = (llvm::AtomicRMWInst::BinOp)binop_tag;
    auto is_valid_fop = is_type_f(continuation->arg(3)->type()) &&
                        (binop == llvm::AtomicRMWInst::BinOp::Xchg || binop == llvm::AtomicRMWInst::BinOp::FAdd || binop == llvm::AtomicRMWInst::BinOp::FSub);
    if (is_type_f(continuation->arg(3)->type()) && !is_valid_fop)
        world().edef(continuation->arg(3), "atomic {} is not supported for float types", binop_tag);
    else if (!is_type_i(continuation->arg(3)->type()) && !is_valid_fop)
        world().edef(continuation->arg(3), "atomic {} is only supported for int types", binop_tag);
    auto ptr = emit(continuation->arg(2));
    auto val = emit(continuation->arg(3));
    u32 order_tag = continuation->arg(4)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(order_tag) && int(order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)order_tag;
    auto scope = continuation->arg(5)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto cont = continuation->arg(6)->as_continuation();
    auto call = irbuilder.CreateAtomicRMW(binop, ptr, val, order, context_->getOrInsertSyncScopeID(scope->as_string()));
    emit_phi_arg(irbuilder, cont->param(1), call);
    return cont;
}

Continuation* CodeGen::emit_atomic_load(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->num_args() == 5 && "required arguments are missing");
    auto ptr = emit(continuation->arg(1));
    u32 tag = continuation->arg(2)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(tag) && int(tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)tag;
    auto scope = continuation->arg(3)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto cont = continuation->arg(4)->as_continuation();
    auto load = irbuilder.CreateLoad(ptr);
    auto align = module().getDataLayout().getABITypeAlignment(ptr->getType()->getPointerElementType());
    load->setAlignment(llvm::MaybeAlign(align));
    load->setAtomic(order, context_->getOrInsertSyncScopeID(scope->as_string()));
    emit_phi_arg(irbuilder, cont->param(1), load);
    return cont;
}

Continuation* CodeGen::emit_atomic_store(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->num_args() == 6 && "required arguments are missing");
    auto ptr = emit(continuation->arg(1));
    auto val = emit(continuation->arg(2));
    u32 tag = continuation->arg(3)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(tag) && int(tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)tag;
    auto scope = continuation->arg(4)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto cont = continuation->arg(5)->as_continuation();
    auto store = irbuilder.CreateStore(val, ptr);
    auto align = module().getDataLayout().getABITypeAlignment(ptr->getType()->getPointerElementType());
    store->setAlignment(llvm::MaybeAlign(align));
    store->setAtomic(order, context_->getOrInsertSyncScopeID(scope->as_string()));
    return cont;
}

Continuation* CodeGen::emit_cmpxchg(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->num_args() == 7 && "required arguments are missing");
    if (!is_type_i(continuation->arg(3)->type()))
        world().edef(continuation->arg(3), "cmpxchg only supported for integer types");
    auto ptr  = emit(continuation->arg(1));
    auto cmp  = emit(continuation->arg(2));
    auto val  = emit(continuation->arg(3));
    u32 order_tag = continuation->arg(4)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(order_tag) && int(order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)order_tag;
    auto scope = continuation->arg(5)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto cont = continuation->arg(6)->as_continuation();
    auto call = irbuilder.CreateAtomicCmpXchg(ptr, cmp, val, order, order, context_->getOrInsertSyncScopeID(scope->as_string()));
    emit_phi_arg(irbuilder, cont->param(1), irbuilder.CreateExtractValue(call, 0));
    emit_phi_arg(irbuilder, cont->param(2), irbuilder.CreateExtractValue(call, 1));
    return cont;
}

Continuation* CodeGen::emit_reserve(llvm::IRBuilder<>&, const Continuation* continuation) {
    world().edef(continuation, "reserve_shared: only allowed in device code"); // TODO debug
    THORIN_UNREACHABLE;
}

Continuation* CodeGen::emit_reserve_shared(llvm::IRBuilder<>& irbuilder, const Continuation* continuation, bool init_undef) {
    assert(continuation->num_args() == 3 && "required arguments are missing");
    if (!continuation->arg(1)->isa<PrimLit>())
        world().edef(continuation->arg(1), "reserve_shared: couldn't extract memory size");
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
    auto call = irbuilder.CreatePointerCast(global, type);
    emit_phi_arg(irbuilder, cont->param(1), call);
    return cont;
}

/*
 * helpers
 */

void CodeGen::create_loop(llvm::IRBuilder<>& irbuilder, llvm::Value* lower, llvm::Value* upper, llvm::Value* increment, llvm::Function* entry, std::function<void(llvm::Value*)> fun) {
    auto head = llvm::BasicBlock::Create(context(), "head", entry);
    auto body = llvm::BasicBlock::Create(context(), "body", entry);
    auto exit = llvm::BasicBlock::Create(context(), "exit", entry);
    // create loop phi and connect init value
    auto loop_counter = llvm::PHINode::Create(irbuilder.getInt32Ty(), 2U, "parallel_loop_phi", head);
    loop_counter->addIncoming(lower, irbuilder.GetInsertBlock());
    // connect head
    irbuilder.CreateBr(head);
    irbuilder.SetInsertPoint(head);
    auto cond = irbuilder.CreateICmpSLT(loop_counter, upper);
    irbuilder.CreateCondBr(cond, body, exit);
    irbuilder.SetInsertPoint(body);

    // add instructions to the loop body
    fun(loop_counter);

    // inc loop counter
    loop_counter->addIncoming(irbuilder.CreateAdd(loop_counter, increment), body);
    irbuilder.CreateBr(head);
    irbuilder.SetInsertPoint(exit);
}

llvm::Value* CodeGen::create_tmp_alloca(llvm::IRBuilder<>& irbuilder, llvm::Type* type, std::function<llvm::Value* (llvm::AllocaInst*)> fun) {
    auto alloca = emit_alloca(irbuilder, type, "tmp_alloca");
    auto size = irbuilder.getInt64(module().getDataLayout().getTypeAllocSize(type));

    irbuilder.CreateLifetimeStart(alloca, size);
    auto result = fun(alloca);
    irbuilder.CreateLifetimeEnd(alloca, size);
    return result;
}

//------------------------------------------------------------------------------

static void get_kernel_configs(Importer& importer,
    const std::vector<Continuation*>& kernels,
    Cont2Config& kernel_config,
    std::function<std::unique_ptr<KernelConfig> (Continuation*, Continuation*)> use_callback)
{
    importer.world().opt();

    auto exported_continuations = importer.world().exported_continuations();
    for (auto continuation : kernels) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto exported : exported_continuations) {
            if (exported->name() == continuation->name())
                imported = exported;
        }
        if (!imported) continue;

        visit_uses(continuation, [&] (Continuation* use) {
            auto config = use_callback(use, imported);
            if (config) {
                auto p = kernel_config.emplace(imported, std::move(config));
                assert_unused(p.second && "single kernel config entry expected");
            }
            return false;
        }, true);

        continuation->destroy_body();
    }
}

static const Continuation* get_alloc_call(const Def* def) {
    // look through casts
    while (auto conv_op = def->isa<ConvOp>())
        def = conv_op->op(0);

    auto param = def->isa<Param>();
    if (!param) return nullptr;

    auto ret = param->continuation();
    if (ret->num_uses() != 1) return nullptr;

    auto use = *(ret->uses().begin());
    auto call = use.def()->isa_continuation();
    if (!call || use.index() == 0) return nullptr;

    auto callee = call->callee();
    if (callee->name() != "anydsl_alloc") return nullptr;

    return call;
}

static uint64_t get_alloc_size(const Def* def) {
    auto call = get_alloc_call(def);
    if (!call) return 0;

    // signature: anydsl_alloc(mem, i32, i64, fn(mem, &[i8]))
    auto size = call->arg(2)->isa<PrimLit>();
    return size ? static_cast<uint64_t>(size->value().get_qu64()) : 0_u64;
}

Backends::Backends(World& world, int opt, bool debug)
    : cuda(world)
    , nvvm(world)
    , opencl(world)
    , amdgpu(world)
    , hls(world)
{
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
        else if (is_passed_to_intrinsic(continuation, Intrinsic::HLS))
            imported = hls.import(continuation)->as_continuation();
        else
            return;

        imported->set_name(continuation->unique_name());
        imported->make_exported();
        continuation->set_name(continuation->unique_name());

        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->set_name(continuation->param(i)->unique_name());

        kernels.emplace_back(continuation);
    });

    // get the GPU kernel configurations
    if (!cuda.world().empty()   ||
        !nvvm.world().empty()   ||
        !opencl.world().empty() ||
        !amdgpu.world().empty()) {
        auto get_gpu_config = [&] (Continuation* use, Continuation* /* imported */) {
            // determine whether or not this kernel uses restrict pointers
            bool has_restrict = true;
            DefSet allocs;
            for (size_t i = LaunchArgs::Num, e = use->num_args(); has_restrict && i != e; ++i) {
                auto arg = use->arg(i);
                if (!arg->type()->isa<PtrType>()) continue;
                auto alloc = get_alloc_call(arg);
                if (!alloc) has_restrict = false;
                auto p = allocs.insert(alloc);
                has_restrict &= p.second;
            }

            auto it_config = use->arg(LaunchArgs::Config)->as<Tuple>();
            if (it_config->op(0)->isa<PrimLit>() &&
                it_config->op(1)->isa<PrimLit>() &&
                it_config->op(2)->isa<PrimLit>()) {
                return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int> {
                    it_config->op(0)->as<PrimLit>()->qu32_value().data(),
                    it_config->op(1)->as<PrimLit>()->qu32_value().data(),
                    it_config->op(2)->as<PrimLit>()->qu32_value().data()
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
        auto get_hls_config = [&] (Continuation* use, Continuation* imported) {
            HLSKernelConfig::Param2Size param_sizes;
            for (size_t i = 3, e = use->num_args(); i != e; ++i) {
                auto arg = use->arg(i);
                auto ptr_type = arg->type()->isa<PtrType>();
                if (!ptr_type) continue;
                auto size = get_alloc_size(arg);
                if (size == 0)
                    world.edef(arg, "array size is not known at compile time");
                auto elem_type = ptr_type->pointee();
                size_t multiplier = 1;
                if (!elem_type->isa<PrimType>()) {
                    if (auto array_type = elem_type->isa<ArrayType>())
                        elem_type = array_type->elem_type();
                }
                if (!elem_type->isa<PrimType>()) {
                    if (auto def_array_type = elem_type->isa<DefiniteArrayType>()) {
                        elem_type = def_array_type->elem_type();
                        multiplier = def_array_type->dim();
                    }
                }
                auto prim_type = elem_type->isa<PrimType>();
                if (!prim_type)
                    world.edef(arg, "only pointers to arrays of primitive types are supported");
                auto num_elems = size / (multiplier * num_bits(prim_type->primtype_tag()) / 8);
                // imported has type: fn (mem, fn (mem), ...)
                param_sizes.emplace(imported->param(i - 3 + 2), num_elems);
            }
            return std::make_unique<HLSKernelConfig>(param_sizes);
        };
        get_kernel_configs(hls, kernels, kernel_config, get_hls_config);
    }

    cpu_cg = std::make_unique<CPUCodeGen>(world, opt, debug);

    if (!nvvm.  world().empty()) nvvm_cg   = std::make_unique<NVVMCodeGen  >(nvvm  .world(), kernel_config,      debug);
    if (!amdgpu.world().empty()) amdgpu_cg = std::make_unique<AMDGPUCodeGen>(amdgpu.world(), kernel_config, opt, debug);

    // TODO
    //if (!cuda.  world().empty()) cuda_cg   = std::make_unique<CUDACodeGen  >(cuda  .world(), kernel_config, opt, debug);
    //if (!opencl.world().empty()) opencl_cg = std::make_unique<OpenCLCodeGen>(opencl.world(), kernel_config, opt, debug);
    //if (!hls.   world().empty()) hls_cg    = std::make_unique<HLSCodeGen   >(hls   .world(), kernel_config, opt, debug);
}

//------------------------------------------------------------------------------

}
