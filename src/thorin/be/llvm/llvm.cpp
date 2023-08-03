#include "thorin/be/llvm/llvm.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map> // TODO don't used std::unordered_*

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
#include "thorin/util/array.h"

namespace thorin::llvm {

CodeGen::CodeGen(
    Thorin& thorin,
    llvm::CallingConv::ID function_calling_convention,
    llvm::CallingConv::ID device_calling_convention,
    llvm::CallingConv::ID kernel_calling_convention,
    int opt, bool debug)
    : thorin::CodeGen(thorin, debug)
    , context_(std::make_unique<llvm::LLVMContext>())
    , module_(std::make_unique<llvm::Module>(world().name(), context()))
    , opt_(opt)
    , dibuilder_(module())
    , function_calling_convention_(function_calling_convention)
    , device_calling_convention_(device_calling_convention)
    , kernel_calling_convention_(kernel_calling_convention)
    , runtime_(std::make_unique<Runtime>(context(), module()))
{}

void CodeGen::optimize() {
    llvm::PassBuilder PB;
    llvm::OptimizationLevel opt_level;

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
        case 0:  opt_level = llvm::OptimizationLevel::O0; break;
        case 1:  opt_level = llvm::OptimizationLevel::O1; break;
        case 2:  opt_level = llvm::OptimizationLevel::O2; break;
        case 3:  opt_level = llvm::OptimizationLevel::O3; break;
        default: opt_level = llvm::OptimizationLevel::Os; break;
    }

    if (opt() == 3) {
        llvm::ModulePassManager module_pass_manager;
        module_pass_manager.addPass(llvm::ModuleInlinerWrapperPass());
        llvm::FunctionPassManager function_pass_manager;
        function_pass_manager.addPass(llvm::ADCEPass());
        module_pass_manager.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(function_pass_manager)));

        module_pass_manager.run(module(), MAM);
    }

    llvm::ModulePassManager builder_passes = PB.buildModuleOptimizationPipeline(opt_level);
    builder_passes.run(module(), MAM);
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
            auto tfn_type = type->as<FnType>();
            llvm::Type* ret = llvm::Type::getVoidTy(context()); // top-level non-returning continuations should be void
            std::vector<llvm::Type*> ops;
            for (auto op : tfn_type->domain()) {
                if (op->isa<MemType>() || op == world().unit_type()) continue;
                ops.push_back(convert(op));
            }

            if (tfn_type->is_returning()) {
                auto ret_ty = tfn_type->return_param_type();
                assert(ret_ty);
                std::vector<llvm::Type*> ret_types;
                for (auto fn_op : ret_ty->types()) {
                    if (fn_op->isa<MemType>() || fn_op == world().unit_type()) continue;
                    ret_types.push_back(convert(fn_op));
                }
                if (ret_types.size() == 0)      ret = llvm::Type::getVoidTy(context());
                else if (ret_types.size() == 1) ret = ret_types.back();
                else                            ret = llvm::StructType::get(context(), ret_types);
            }

            assert(ret);

            if (type->tag() == Node_FnType) {
                auto llvm_type = llvm::FunctionType::get(ret, ops, false);
                return types_[type] = llvm_type;
            }

            auto env_type = convert(Closure::environment_type(world()));
            ops.push_back(env_type);
            auto fn_type = llvm::FunctionType::get(ret, ops, false);
            auto ptr_type = llvm::PointerType::get(fn_type, 0);
            llvm_type = llvm::StructType::get(context(), { ptr_type, env_type });
            return types_[type] = llvm_type;
        }
        case Node_ReturnType: {
            auto ret_t = type->as<ReturnType>();
            auto tuple_t = world().tuple_type({ ret_t->mangle_for_codegen(), world().definite_array_type(world().type_qu8(), 200) });
            return types_[type] = convert(world().ptr_type(tuple_t));
        }

        case Node_StructType: {
            auto struct_type = type->as<StructType>();
            auto llvm_struct = llvm::StructType::create(context());

            // important: memoize before recursing into element types to avoid endless recursion
            assert(!types_.contains(struct_type) && "type already converted");
            types_[struct_type] = llvm_struct;

            Array<llvm::Type*> llvm_types(struct_type->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(struct_type->types()[i]);
            llvm_struct->setBody(llvm_ref(llvm_types));
            return llvm_struct;
        }

        case Node_TupleType: {
            auto tuple = type->as<TupleType>();
            Array<llvm::Type*> llvm_types(tuple->num_ops());
            for (size_t i = 0, e = llvm_types.size(); i != e; ++i)
                llvm_types[i] = convert(tuple->types()[i]);
            llvm_type = llvm::StructType::get(context(), llvm_ref(llvm_types));
            return types_[tuple] = llvm_type;
        }

        case Node_VariantType: {
            auto variant_type = type->as<VariantType>();
            assert(type->num_ops() > 0);
            // Max alignment/size constraints respectively in the variant type alternatives dictate the ones to use for the overall type
            size_t max_align = 0, max_size = 0;

            auto layout = module().getDataLayout();
            llvm::Type* max_align_type;
            for (auto op : variant_type->types()) {
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

    llvm_type = llvm::FixedVectorType::get(llvm_type, vector_length(type));
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

void CodeGen::emit_stream(std::ostream& stream) {
    llvm::raw_os_ostream llvm_stream(stream);
    emit_module().second->print(llvm_stream, nullptr);
}

std::pair<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
CodeGen::emit_module() {
    if (debug()) {
        module().addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
        // Darwin only supports dwarf2
        if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin())
            module().addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
        dicompile_unit_ = dibuilder_.createCompileUnit(llvm::dwarf::DW_LANG_C, dibuilder_.createFile(world().name(), llvm::StringRef()), "Impala", opt() > 0, llvm::StringRef(), 0);
    }

    for (auto&& [_, def] : world().externals()) {
        if (auto global = def->isa<Global>()) {
            emit(global);
        }
    }

    ScopesForest(world()).for_each([&] (const Scope& scope) { emit_scope(scope); });

    if (debug()) dibuilder_.finalize();

#if THORIN_ENABLE_RV
    for (auto [width, fct, call] : vec_todo_)
        emit_vectorize(width, fct, call);
    vec_todo_.clear();

    rv::lowerIntrinsics(module());
#endif

    verify();
    optimize();

    // We need to delete the runtime at this point, since the ownership of
    // the context and module is handed away.
    runtime_.reset();
    return std::pair { std::move(context_), std::move(module_) };
}

llvm::Function* CodeGen::emit_fun_decl(Continuation* continuation) {
    std::string name = world().is_external(continuation) ? continuation->name() : continuation->unique_name();
    auto f = llvm::cast<llvm::Function>(module().getOrInsertFunction(name, convert_fn_type(continuation)).getCallee()->stripPointerCasts());
    if (machine_) {
        f->addFnAttr("target-cpu", machine().getTargetCPU());
        f->addFnAttr("target-features", machine().getTargetFeatureString());
    }

#ifdef _MSC_VER
    // set dll storage class for MSVC
    if (!entry_ && llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()) {
        if (continuation->is_imported()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        } else if (continuation->is_exported()) {
            f->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }
#endif

    // set linkage
    if (world().is_external(continuation))
        f->setLinkage(llvm::Function::ExternalLinkage);
    else
        f->setLinkage(llvm::Function::InternalLinkage);

    // set calling convention
    if (continuation->is_exported()) {
        f->setCallingConv(kernel_calling_convention_);
        emit_fun_decl_hook(continuation, f);
    } else {
        if (continuation->cc() == CC::Device)
            f->setCallingConv(device_calling_convention_);
        else
            f->setCallingConv(function_calling_convention_);
    }

    return f;
}

llvm::Function* CodeGen::prepare(const Scope& scope) {
    auto fct = llvm::cast<llvm::Function>(emit(scope.entry()));

    discope_ = dicompile_unit_;
    if (debug()) {
        auto file = entry_->loc().file;
        auto src_file = llvm::sys::path::filename(file);
        auto src_dir = llvm::sys::path::parent_path(file);
        auto difile = dibuilder_.createFile(src_file, src_dir);
        auto disub_program = dibuilder_.createFunction(
            discope_, fct->getName(), fct->getName(), difile, entry_->loc().begin.row,
            dibuilder_.createSubroutineType(dibuilder_.getOrCreateTypeArray(llvm::ArrayRef<llvm::Metadata*>())),
            entry_->loc().begin.row,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition | (opt() > 0 ? llvm::DISubprogram::SPFlagOptimized : llvm::DISubprogram::SPFlagZero));
        fct->setSubprogram(disub_program);
        discope_ = disub_program;
    }

    has_alloca_ = false;
    potential_tailcalls_.clear();

    return_buf_ = nullptr;
    entry_prelude_.reset();
    entry_prelude_end_.reset();
    if (entry_->is_returning()) {
        auto ret_p = entry_->ret_param();
        auto ret_t = entry_->type()->return_param_type();
        assert(ret_p && ret_t);
        convert(ret_t);
        bool leaks = false;
        for (auto use: ret_p->uses()) {
            auto app = use.def()->isa<App>();
            if (app && use.index() == App::CALLEE_POSITION)
                continue; // ret(...)
            if (app && app->callee()->type()->as<FnType>()->ret_param_index() == (int) use.index() - App::ARGS_START_POSITION)
                continue; // foo(a, b, c, ... ret) where arg n-1 is of type ret[...]
            leaks = true;
            break;
        }
        if (leaks) {
            entry_prelude_ = std::make_optional<>(std::pair(llvm::BasicBlock::Create(context(), "entry_prelude_", fct), std::make_unique<llvm::IRBuilder<>>(context())));
            entry_prelude_->second->SetInsertPoint(entry_prelude_->first);
            entry_prelude_end_ = std::make_optional<>(std::pair(llvm::BasicBlock::Create(context(), "entry_prelude_end_", fct), std::make_unique<llvm::IRBuilder<>>(context())));
            entry_prelude_end_->second->SetInsertPoint(entry_prelude_end_->first);
            auto if_setjmp = std::make_optional<>(std::pair(llvm::BasicBlock::Create(context(), "if_setjmp", fct), std::make_unique<llvm::IRBuilder<>>(context())));
            if_setjmp->second->SetInsertPoint(if_setjmp->first);
            auto& builder = *entry_prelude_->second;

            auto returned_type = ret_t->mangle_for_codegen();
            auto jmp_buf_type = world().definite_array_type(world().type_qu8(), 200);

            return_buf_ = emit_alloca(builder, convert(ret_t)->getPointerElementType(), "return_buf");
            auto ret_value_buf = builder.CreateConstGEP2_32(convert(ret_t)->getPointerElementType(), return_buf_, 0, 0);
            ret_value_buf->setName("return_value_buf");
            auto jmp_buf =  builder.CreateConstGEP2_32(convert(ret_t)->getPointerElementType(), return_buf_, 0, 1);
            jmp_buf->setName("jmp_buf");
            defs_[entry_->ret_param()] = return_buf_;

            auto fn_type = llvm::FunctionType::get(llvm::IntegerType::get(context(), 32), { llvm::PointerType::get(llvm::IntegerType::get(context(), 8), 0) }, false);
            auto fn = llvm::cast<llvm::Function>(module().getOrInsertFunction("setjmp", fn_type).getCallee()->stripPointerCasts());

            auto result = entry_prelude_->second->CreateCall(fn, { entry_prelude_->second->CreatePointerCast(jmp_buf, llvm::PointerType::get(llvm::IntegerType::get(context(), 8), 0)) });
            auto branch = entry_prelude_->second->CreateICmpEQ(result, llvm::ConstantInt::get(result->getType(), 0));
            entry_prelude_->second->CreateCondBr(branch, entry_prelude_end_->first, if_setjmp->first);

            if (returned_type != world().unit_type()) {
                auto loaded_ret_value = if_setjmp->second->CreateLoad(convert(returned_type), ret_value_buf);
                if_setjmp->second->CreateRet(loaded_ret_value);
            } else {
                if_setjmp->second->CreateRetVoid();
            }
        }
    }

    return fct;
}

void CodeGen::prepare(Continuation* cont, llvm::Function* fct) {
    // map all bb-like continuations to llvm bb stubs and handle params/phis
    auto bb = llvm::BasicBlock::Create(context(), cont->name().c_str(), fct);
    auto [i, succ] = cont2bb_.emplace(cont, std::pair(bb, std::make_unique<llvm::IRBuilder<>>(context())));
    assert(succ);
    auto& irbuilder = *i->second.second;
    irbuilder.SetInsertPoint(bb);

    if (debug())
        irbuilder.SetCurrentDebugLocation(llvm::DILocation::get(discope_->getContext(), cont->loc().begin.row, cont->loc().begin.col, discope_));

    if (entry_ == cont) {
        auto arg = fct->arg_begin();
        for (auto param : entry_->params()) {
            if (is_mem(param) || is_unit(param)) {
                defs_[param] = nullptr;
            } else if (param->order() == 0) {
                auto argv = &*arg;
                auto value = map_param(fct, argv, param);
                if (value == argv) {
                    arg->setName(param->unique_name()); // use param
                    defs_[param] = &*arg++;
                } else {
                    defs_[param] = value;               // use provided value
                }
            }
        }

        if (entry_prelude_end_)
            entry_prelude_end_->second->CreateBr(cont2bb(entry_));
    } else {
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param)) {
                defs_[param] = nullptr;
            } else {
                // do not bother reserving anything (the 0 below) - it's a tiny optimization nobody cares about
                auto phi = irbuilder.CreatePHI(convert(param->type()), 0, param->name().c_str());
                defs_[param] = phi;
            }
        }
    }
}

void CodeGen::finalize(const Scope&) {
    std::vector<const Def*> to_remove;
    for (auto& [def, value] : defs_) {
        // These do not have scope dependencies in Thorin, but they translate to LLVM alloca loads
        // which should never be reused outside of the function they were defined in, so we erase them
        if (auto variant = def->isa<Variant>(); variant && !variant->value()->has_dep(Dep::Param))
            to_remove.push_back(def);
    }
    if (!has_alloca_) for (auto call : potential_tailcalls_) {
        call->setTailCall(true);
        // call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    }
    for (auto& def : to_remove)
        defs_.erase(def);
}

std::vector<llvm::Value*> CodeGen::split_values(llvm::IRBuilder<>& irbuilder, Types domain, llvm::Value* value) {
    size_t n = 0;
    for (auto t : domain) {
        if (t == world().unit_type() || t->isa<MemType>())
            continue;
        n++;
    }

    switch (n) {
        case 0: return {};
        case 1: return { value };
        default: {
            std::vector<llvm::Value*> values;
            values.reserve(n);
            for (size_t i = 0; i < n; i++) {
                values[i] = irbuilder.CreateExtractValue(value, i);
            }
            return values;
        }
    }
}

llvm::Value* CodeGen::emit_call(llvm::IRBuilder<>& irbuilder, const Def* callee, std::vector<llvm::Value*>& args) {
    if (callee == entry_->ret_param()) { // normal return
        std::vector<llvm::Type *> types;
        for (auto val : args)
            types.emplace_back(val->getType());
        switch (args.size()) {
            case 0:  irbuilder.CreateRetVoid();      break;
            case 1:  irbuilder.CreateRet(args[0]); break;
            default: {
                llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context(), types));

                for (size_t i = 0, e = args.size(); i != e; ++i)
                    agg = irbuilder.CreateInsertValue(agg, args[i], { unsigned(i) });

                irbuilder.CreateRet(agg);
            }
        }
        return nullptr;
    } else if (auto return_point = callee->isa<ReturnPoint>()) { // for direct-style calls, just forward to the destination
        return emit_call(irbuilder, return_point->continuation(), args);
    } else if (auto ret_t = callee->type()->isa<ReturnType>()) { // 'far' return
        auto returned_type = ret_t->mangle_for_codegen();
        auto jmp_buf_type = world().definite_array_type(world().type_qu8(), 200);
        auto return_buf = emit(callee);
        auto ret_value_buf = irbuilder.CreateConstGEP2_32(convert(ret_t)->getPointerElementType(), return_buf, 0, 0);
        ret_value_buf->setName("return_value_buf");
        auto jmp_buf =  irbuilder.CreateConstGEP2_32(convert(ret_t)->getPointerElementType(), return_buf, 0, 1);
        jmp_buf->setName("jmp_buf");

        auto fn_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context()), { llvm::PointerType::get(llvm::IntegerType::get(context(), 8), 0), llvm::IntegerType::get(context(), 32) }, false);
        auto fn = llvm::cast<llvm::Function>(module().getOrInsertFunction("longjmp", fn_type).getCallee()->stripPointerCasts());

        std::vector<llvm::Type *> types;
        for (auto arg : args)
            types.emplace_back(arg->getType());

        llvm::Value* ret_value = nullptr;
        switch (args.size()) {
            case 0: break;
            case 1: ret_value = args[0]; break;
            default:
                llvm::Value* agg = llvm::UndefValue::get(llvm::StructType::get(context(), types));

                for (size_t i = 0, e = args.size(); i != e; ++i)
                    agg = irbuilder.CreateInsertValue(agg, args[i], { unsigned(i) });

                ret_value = agg;
                break;
        }

        if (ret_value)
            irbuilder.CreateStore(ret_value, ret_value_buf);
        auto call = irbuilder.CreateCall(fn, { irbuilder.CreatePointerCast(jmp_buf, llvm::PointerType::get(llvm::IntegerType::get(context(), 8), 0)), emit(world().literal_ps32(1, {})) });
        return call;
    } else if (callee->isa<Bottom>()) {
        irbuilder.CreateUnreachable();
        return nullptr;
    } else if (auto cont = callee->isa_nom<Continuation>(); cont && scope_->contains(cont) && cont != entry_) {
        assert(cont->is_basicblock());
        size_t j = 0, i = 0;
        for (auto t: cont->type()->domain()) {
            i++;
            if (t->isa<MemType>() || t == world().unit_type())
                continue;
            emit_phi_arg(irbuilder, cont->param(i - 1), args[j++]);
        }
        irbuilder.CreateBr(cont2bb(cont));
        return nullptr;
    } else if (auto closure_t = callee->type()->isa<ClosureType>()) {
        auto closure = emit(callee);
        auto fnt = world().fn_type(concat(closure_t->types(), closure_t->as<Type>()));
        args.push_back(closure);
        auto func = irbuilder.CreateExtractValue(closure, 0);
        auto call = irbuilder.CreateCall(llvm::cast<llvm::FunctionType>(convert(fnt)), irbuilder.CreatePointerCast(func, convert(world().ptr_type(fnt))), args);
        return call;
    } else if (callee->type()->tag() == Node_FnType) {
        auto call = irbuilder.CreateCall(llvm::cast<llvm::Function>(emit(callee)), args);
        if (cont->is_exported())
            call->setCallingConv(kernel_calling_convention_);
        else if (cont->cc() == CC::Device)
            call->setCallingConv(device_calling_convention_);
        else
            call->setCallingConv(function_calling_convention_);
        return call;
    }

    THORIN_UNREACHABLE;
}

void CodeGen::emit_epilogue(Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();

    auto& [bb, ptr_irbuilder] = cont2bb_[continuation];
    auto& irbuilder = *ptr_irbuilder;

    llvm::Value* call_instr = nullptr;

    if (body->callee() == world().branch()) {
        auto mem = body->arg(0);
        emit_unsafe(mem);

        auto cond = emit(body->arg(1));
        auto tbb = cont2bb(body->arg(2)->as_nom<Continuation>());
        auto fbb = cont2bb(body->arg(3)->as_nom<Continuation>());
        irbuilder.CreateCondBr(cond, tbb, fbb);
    } else if (body->callee()->isa<Continuation>() && body->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
        auto mem = body->arg(0);
        emit_unsafe(mem);

        auto val = emit(body->arg(1));
        auto otherwise_bb = cont2bb(body->arg(2)->as_nom<Continuation>());
        auto match = irbuilder.CreateSwitch(val, otherwise_bb, body->num_args() - 3);
        for (size_t i = 3; i < body->num_args(); i+=2) {
            auto case_const = llvm::cast<llvm::ConstantInt>(emit(body->arg(i)));
            auto case_bb    = cont2bb(body->arg(i + 1)->as_nom<Continuation>());
            match->addCase(case_const, case_bb);
        }
    } else if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->is_intrinsic()) {
        auto args = emit_intrinsic(irbuilder, continuation);
        call_instr = emit_call(irbuilder, body->arg(callee->type()->ret_param_index()), args);
    } else {
        // plain continuation call: we can just emit all the arguments
        std::vector<llvm::Value*> args;
        const Def* ret_arg = nullptr;
        for (auto arg : body->args()) {
            if (arg->order() == 0) {
                if (auto val = emit_unsafe(arg))
                    args.push_back(val);
            } else {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        call_instr = emit_call(irbuilder, body->callee(), args);
        // todo potential_tailcalls_ fast-path ?
        if (auto codom = body->callee()->type()->as<FnType>()->codomain()) {
            assert(call_instr && "returning calls always involve one of those");
            assert(ret_arg && "we need a return argument too!");
            auto ret_args = split_values(irbuilder, *codom, call_instr);
            call_instr = emit_call(irbuilder, ret_arg, ret_args);
        }
    }

    if (call_instr) {
        // we need to add a dummy return terminator if the last instruction is a call
        if (entry_->type()->is_returning()) {
            auto entry_return_t = entry_->type()->return_param_type()->mangle_for_codegen();
            if (entry_return_t != world().unit_type()) {
                irbuilder.CreateRet(llvm::UndefValue::get(convert(entry_return_t)));
            } else
                irbuilder.CreateRetVoid();
        } else
            irbuilder.CreateRetVoid();
    }

    // new insert point is just before the terminator for all other instructions we have to add later on
    irbuilder.SetInsertPoint(bb->getTerminator());
}

llvm::Value* CodeGen::emit_constant(const Def* def) {
    auto irbuilder = llvm::IRBuilder(context());
    return emit_builder(irbuilder, def);
}

llvm::Value* CodeGen::emit_bb(BB& bb, const Def* def) {
    auto& irbuilder = *bb.second;
    return emit_builder(irbuilder, def);
}

llvm::Value* CodeGen::emit_builder(llvm::IRBuilder<>& irbuilder, const Def* def) {
    // TODO
    //if (debug())
        //irbuilder.SetCurrentDebugLocation(llvm::DILocation::get(discope_->getContext(), def->loc().begin.row, def->loc().begin.col, discope_));

    if (false) {}
    else if (auto load = def->isa<Load>())           return emit_load(irbuilder, load);
    else if (auto store = def->isa<Store>())         return emit_store(irbuilder, store);
    else if (auto lea = def->isa<LEA>())             return emit_lea(irbuilder, lea);
    else if (auto assembly = def->isa<Assembly>())   return emit_assembly(irbuilder, assembly);
    else if (def->isa<Enter>())                      return emit_unsafe(def->op(0));
    else if (auto bin = def->isa<BinOp>()) {
        llvm::Value* lhs = emit(bin->lhs());
        llvm::Value* rhs = emit(bin->rhs());
        auto name = bin->name();

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
        } else if (auto arithop = bin->isa<ArithOp>()) {
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
            } else if (is_type_s(type) || is_type_bool(type)) {
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
            } else if (is_type_u(type) || is_type_bool(type)) {
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
    } else if (auto mathop = def->isa<MathOp>()) {
        return emit_mathop(irbuilder, mathop);
    } else if (auto conv = def->isa<ConvOp>()) {
        auto from = emit(conv->from());
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();
        auto to = convert(dst_type);

        if ((src_type->isa<PtrType>() && dst_type->isa<PtrType>()) || (src_type->isa<ReturnType>() && dst_type->isa<ReturnType>())) {
            return irbuilder.CreatePointerCast(from, to);
        } else if (src_type->isa<PtrType>() || src_type->isa<ReturnType>()) {
            assert(is_type_i(dst_type) || is_type_bool(dst_type));
            return irbuilder.CreatePtrToInt(from, to);
        } else if (dst_type->isa<PtrType>() || dst_type->isa<ReturnType>()) {
            assert(is_type_i(src_type) || is_type_bool(src_type));
            return irbuilder.CreateIntToPtr(from, to);
        }

        if (conv->isa<Cast>()) {
            auto src = src_type->as<PrimType>();
            auto dst = dst_type->as<PrimType>();

            if (is_type_f(src) && is_type_f(dst)) {
                assert(num_bits(src->primtype_tag()) != num_bits(dst->primtype_tag()));
                return irbuilder.CreateFPCast(from, to);
            } else if (is_type_f(src)) {
                if (is_type_s(dst))
                    return irbuilder.CreateFPToSI(from, to);
                return irbuilder.CreateFPToUI(from, to);
            } else if (is_type_f(dst)) {
                if (is_type_s(src))
                    return irbuilder.CreateSIToFP(from, to);
                return irbuilder.CreateUIToFP(from, to);
            } else if (num_bits(src->primtype_tag()) > num_bits(dst->primtype_tag())) {
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
        } else if (conv->isa<Bitcast>()) {
            return emit_bitcast(irbuilder, conv->from(), dst_type);
        }
    } else if (auto select = def->isa<Select>()) {
        if (def->type()->isa<FnType>())
            return nullptr;

        llvm::Value* cond = emit(select->cond());
        llvm::Value* tval = emit(select->tval());
        llvm::Value* fval = emit(select->fval());
        return irbuilder.CreateSelect(cond, tval, fval);
    } else if (auto align_of = def->isa<AlignOf>()) {
        auto type = convert(align_of->of());
        return irbuilder.getInt64(module().getDataLayout().getABITypeAlignment(type));
    } else if (auto size_of = def->isa<SizeOf>()) {
        auto type = convert(size_of->of());
        return irbuilder.getInt64(module().getDataLayout().getTypeAllocSize(type));
    } else if (auto array = def->isa<DefiniteArray>()) {
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
            auto gep = irbuilder.CreateInBoundsGEP(type, alloca, args, op->name().c_str());
            irbuilder.CreateStore(emit(op), gep);
        }

        return irbuilder.CreateLoad(alloca->getAllocatedType(), alloca);
    } else if (auto array = def->isa<IndefiniteArray>()) {
        return llvm::UndefValue::get(convert(array->type()));
    } else if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>() || def->isa<Closure>());
        if (is_unit(agg)) return nullptr;

        if (def->isa<StructAgg>() || def->isa<Vector>()) {
            // Try to emit it as a constant first
            Array<llvm::Constant*> consts(agg->num_ops());
            bool all_consts = true;
            for (size_t i = 0, n = consts.size(); i != n; ++i) {
                consts[i] = llvm::dyn_cast<llvm::Constant>(emit(agg->op(i)));
                if (!consts[i]) {
                    all_consts = false;
                    break;
                }
            }
            if (all_consts) {
                if (def->isa<StructAgg>())
                    return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(convert(agg->type())), llvm_ref(consts));
                else
                    return llvm::ConstantVector::get(llvm_ref(consts));
            }
        }

        llvm::Value* llvm_agg = llvm::UndefValue::get(convert(agg->type()));
        if (def->isa<Vector>()) {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder.CreateInsertElement(llvm_agg, emit(agg->op(i)), irbuilder.getInt32(i));
        } else if (auto closure = def->isa<Closure>()) {
            auto closure_fn = irbuilder.CreatePointerCast(emit(closure->fn()), llvm_agg->getType()->getStructElementType(0));
            auto val = closure->env();
            llvm::Value* env = nullptr;
            if (is_type_unit(val->type())) {
                env = emit(world().bottom(Closure::environment_type(world())));
            } else {
                env = emit_builder(irbuilder, world().cast(Closure::environment_type(world()), val));
            }
            llvm_agg = irbuilder.CreateInsertValue(llvm_agg, closure_fn, 0);
            llvm_agg = irbuilder.CreateInsertValue(llvm_agg, env, 1);
        } else {
            for (size_t i = 0, e = agg->num_ops(); i != e; ++i)
                llvm_agg = irbuilder.CreateInsertValue(llvm_agg, emit(agg->op(i)), { unsigned(i) });
        }

        return llvm_agg;
    } else if (auto cell = def->isa<Cell>()) {
        auto alloc = cell->is_heap_allocated() ? emit_alloc(irbuilder, cell->contents()->type(), nullptr) : emit_alloca(irbuilder, convert(cell->contents()->type()), "cell");
        defs_[cell] = alloc;
        irbuilder.CreateStore(emit(cell->contents()), alloc);
        return irbuilder.CreatePtrToInt(alloc, convert(Closure::environment_type(world())));
    } else if (auto aggop = def->isa<AggOp>()) {
        auto llvm_agg = emit_unsafe(aggop->agg());
        auto llvm_idx = emit(aggop->index());

        bool mem = false;
        if (auto tt = aggop->agg()->type()->isa<TupleType>(); tt && tt->op(0)->isa<MemType>()) mem = true;

        auto copy_to_alloca = [&] () {
            world().wdef(def, "slow: alloca and loads/stores needed for aggregate '{}'", def);
            auto alloca = emit_alloca(irbuilder, llvm_agg->getType(), aggop->name());
            irbuilder.CreateStore(llvm_agg, alloca);

            llvm::Value* args[2] = { irbuilder.getInt64(0), llvm_idx };
            auto gep = irbuilder.CreateInBoundsGEP(llvm_agg->getType(), alloca, args);
            return std::make_pair(alloca, gep);
        };
        auto copy_to_alloca_or_global = [&] () -> llvm::Value* {
            if (auto constant = llvm::dyn_cast<llvm::Constant>(llvm_agg)) {
                auto global = llvm::cast<llvm::GlobalVariable>(module().getOrInsertGlobal(aggop->agg()->unique_name().c_str(), llvm_agg->getType()));
                global->setLinkage(llvm::GlobalValue::InternalLinkage);
                global->setInitializer(constant);
                return irbuilder.CreateInBoundsGEP(llvm_agg->getType(), global, { irbuilder.getInt64(0), llvm_idx });
            }
            return copy_to_alloca().second;
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (aggop->agg()->type()->isa<ArrayType>()) {
                auto alloca_or_global = copy_to_alloca_or_global();
                return irbuilder.CreateLoad(alloca_or_global->getType()->getPointerElementType(), alloca_or_global);
            } else if (extract->agg()->type()->isa<VectorType>()) {
                return irbuilder.CreateExtractElement(llvm_agg, llvm_idx);
            }

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
            return irbuilder.CreateLoad(p.first->getType(), p.first);
        } else if (insert->agg()->type()->isa<VectorType>()) {
            return irbuilder.CreateInsertElement(llvm_agg, emit(aggop->as<Insert>()->value()), llvm_idx);
        }
        // tuple/struct
        return irbuilder.CreateInsertValue(llvm_agg, value, {primlit_value<unsigned>(aggop->index())});
    } else if (auto variant_index = def->isa<VariantIndex>()) {
        auto llvm_value = emit(variant_index->op(0));
        auto tag_value = irbuilder.CreateExtractValue(llvm_value, { 1 });
        return irbuilder.CreateIntCast(tag_value, convert(variant_index->type()), false);
    } else if (auto variant_extract = def->isa<VariantExtract>()) {
        auto variant_value = variant_extract->op(0);
        auto llvm_value    = emit(variant_value);
        auto target_type   = variant_value->type()->op(variant_extract->index())->as<Type>();
        if (is_type_unit(target_type))
            return nullptr;

        auto payload_value = irbuilder.CreateExtractValue(llvm_value, { 0 });
        return create_tmp_alloca(irbuilder, payload_value->getType(), [&] (llvm::AllocaInst* alloca) {
            irbuilder.CreateStore(payload_value, alloca);
            auto addr_space = alloca->getType()->getPointerAddressSpace();
            auto payload_addr = irbuilder.CreatePointerCast(alloca, llvm::PointerType::get(convert(target_type), addr_space));
            return irbuilder.CreateLoad(payload_addr->getType()->getPointerElementType(), payload_addr);
        });
    } else if (auto variant_ctor = def->isa<Variant>()) {
        auto llvm_type = convert(variant_ctor->type());
        auto tag_value = irbuilder.getIntN(llvm_type->getStructElementType(1)->getScalarSizeInBits(), variant_ctor->index());

        //Unit type variants can be emitted as constants.
        if (is_type_unit(variant_ctor->op(0)->type())) {
            return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(llvm_type), {llvm::UndefValue::get(llvm_type->getStructElementType(0)), tag_value});
        }

        return create_tmp_alloca(irbuilder, llvm_type, [&] (llvm::AllocaInst* alloca) {
            auto tag_addr = irbuilder.CreateInBoundsGEP(llvm_type, alloca, { irbuilder.getInt32(0), irbuilder.getInt32(1) });
            irbuilder.CreateStore(tag_value, tag_addr);

            // Do not store anything if the payload is unit
            if (!is_type_unit(variant_ctor->op(0)->type())) {
                auto payload_value = emit(variant_ctor->op(0));
                auto payload_addr = irbuilder.CreatePointerCast(
                    irbuilder.CreateInBoundsGEP(llvm_type, alloca, { irbuilder.getInt32(0), irbuilder.getInt32(0) }),
                    llvm::PointerType::get(payload_value->getType(), alloca->getType()->getPointerAddressSpace()));
                irbuilder.CreateStore(payload_value, payload_addr);
            }
            return irbuilder.CreateLoad(alloca->getAllocatedType(), alloca);
        });
    } else if (auto primlit = def->isa<PrimLit>()) {
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
    } else if (auto bottom = def->isa<Bottom>()) {
        return llvm::UndefValue::get(convert(bottom->type()));
    } else if (auto alloc = def->isa<Alloc>()) {
        emit_unsafe(alloc->mem());
        return emit_alloc(irbuilder, alloc->alloced_type(), alloc->extra());
    } else if (auto slot = def->isa<Slot>()) {
        return emit_alloca(irbuilder, convert(slot->type()->as<PtrType>()->pointee()), slot->unique_name());
    } else if (auto vector = def->isa<Vector>()) {
        llvm::Value* vec = llvm::UndefValue::get(convert(vector->type()));
        for (size_t i = 0, e = vector->num_ops(); i != e; ++i)
            vec = irbuilder.CreateInsertElement(vec, emit(vector->op(i)), emit(world().literal_pu32(i, vector->loc())));

        return vec;
    } else if (auto global = def->isa<Global>()) {
        return emit_global(global);
    }

    THORIN_UNREACHABLE;
}

void CodeGen::emit_phi_arg(llvm::IRBuilder<>& irbuilder, const Param* param, llvm::Value* value) {
    assert(defs_[param]);
    llvm::cast<llvm::PHINode>(defs_[param])->addIncoming(value, irbuilder.GetInsertBlock());
}

/*
 * emit: special overridable methods
 */

llvm::Value* CodeGen::emit_alloc(llvm::IRBuilder<>& irbuilder, const Type* type, const Def* extra) {
    auto llvm_malloc = runtime_->get(*this, get_alloc_name().c_str());
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
    alloca->setAlignment(layout.getABITypeAlign(type));
    has_alloca_ = true;
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
    if (auto continuation = global->init()->isa_nom<Continuation>())
        val = emit(continuation);
    else {
        auto llvm_type = convert(global->alloced_type());
        auto var = llvm::cast<llvm::GlobalVariable>(module().getOrInsertGlobal(global->is_external() ? global->name().c_str() : global->unique_name().c_str(), llvm_type));
        var->setConstant(!global->is_mutable());

        if (global->init()->isa<Bottom>()) {
            if (global->is_external())
                var->setExternallyInitialized(true);
            else
                var->setInitializer(llvm::Constant::getNullValue(llvm_type)); // HACK
        } else
            var->setInitializer(llvm::cast<llvm::Constant>(emit(global->init())));

        if (global->is_external()) {
            var->setAlignment(llvm::Align(4));
            var->setDSOLocal(true);
            var->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::None);
        } else
            var->setLinkage(llvm::GlobalValue::InternalLinkage);
        val = var;
    }
    return val;
}

llvm::GlobalVariable* CodeGen::emit_global_variable(llvm::Type* type, const std::string& name, unsigned addr_space, bool init_undef) {
    auto init = init_undef ? llvm::UndefValue::get(type) : llvm::Constant::getNullValue(type);
    return new llvm::GlobalVariable(module(), type, false, llvm::GlobalValue::InternalLinkage, init, name, nullptr, llvm::GlobalVariable::NotThreadLocal, addr_space);
}

static inline std::string intrinsic_suffix(const thorin::PrimType* type) {
    switch (type->primtype_tag()) {
        case PrimType_qf32:
        case PrimType_pf32:
            return "f32";
        case PrimType_qf64:
        case PrimType_pf64:
            return "f64";
        default:
            THORIN_UNREACHABLE;
    }
}

llvm::Value* CodeGen::emit_mathop(llvm::IRBuilder<>& irbuilder, const MathOp* mathop) {
    static const std::unordered_map<MathOpTag, std::string> intrinsic_prefixes = {
        { MathOp_copysign, "llvm.copysign." },
        { MathOp_fabs,     "llvm.fabs." },
        { MathOp_fmin,     "llvm.minnum." },
        { MathOp_fmax,     "llvm.maxnum." },
        { MathOp_round,    "llvm.round." },
        { MathOp_floor,    "llvm.floor." },
        { MathOp_ceil,     "llvm.ceil." },
        { MathOp_cos,      "llvm.cos." },
        { MathOp_sin,      "llvm.sin." },
        { MathOp_sqrt,     "llvm.sqrt." },
        { MathOp_pow,      "llvm.pow." },
        { MathOp_exp,      "llvm.exp." },
        { MathOp_exp2,     "llvm.exp2." },
        { MathOp_log,      "llvm.log." },
        { MathOp_log2,     "llvm.log2." },
        { MathOp_log10,    "llvm.log10." }
    };
    static const std::unordered_map<MathOpTag, std::string> math_function_prefix = {
        { MathOp_tan,   "tan" },
        { MathOp_acos,  "acos" },
        { MathOp_asin,  "asin" },
        { MathOp_atan,  "atan" },
        { MathOp_atan2, "atan2" },
        { MathOp_cbrt,  "cbrt" },
    };

    std::string function_name;
    if (auto it = intrinsic_prefixes.find(mathop->mathop_tag()); it != intrinsic_prefixes.end()) {
        function_name = it->second + intrinsic_suffix(mathop->type());
    } else if (auto it = math_function_prefix.find(mathop->mathop_tag()); it != math_function_prefix.end()) {
        auto primtype_tag = mathop->type()->as<PrimType>()->primtype_tag();
        function_name = it->second + ((primtype_tag == PrimType_qf32 || primtype_tag == PrimType_pf32) ? "f" : "");
    } else
        THORIN_UNREACHABLE;

    return call_math_function(irbuilder, mathop, function_name);
}

llvm::Value* CodeGen::call_math_function(llvm::IRBuilder<>& irbuilder, const MathOp* mathop, const std::string& function_name) {
    if (mathop->num_ops() == 1) {
        // Unary mathematical operations
        auto arg = emit(mathop->op(0));
        auto fn_type = llvm::FunctionType::get(arg->getType(), { arg->getType() }, false);
        auto fn = llvm::cast<llvm::Function>(module().getOrInsertFunction(function_name, fn_type).getCallee()->stripPointerCasts());
        if (machine_) {
            fn->addFnAttr("target-cpu", machine().getTargetCPU());
            fn->addFnAttr("target-features", machine().getTargetFeatureString());
        }
        return irbuilder.CreateCall(fn, { arg });
    } else if (mathop->num_ops() == 2) {
        // Binary mathematical operations
        auto left = emit(mathop->op(0));
        auto right = emit(mathop->op(1));
        auto fn_type = llvm::FunctionType::get(left->getType(), { left->getType(), right->getType() }, false);
        auto fn = llvm::cast<llvm::Function>(module().getOrInsertFunction(function_name, fn_type).getCallee()->stripPointerCasts());
        if (machine_) {
            fn->addFnAttr("target-cpu", machine().getTargetCPU());
            fn->addFnAttr("target-features", machine().getTargetFeatureString());
        }
        return irbuilder.CreateCall(fn, { left, right });
    }
    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_load(llvm::IRBuilder<>& irbuilder, const Load* load) {
    emit_unsafe(load->mem());
    auto ptr = emit(load->ptr());
    auto result = irbuilder.CreateLoad(ptr->getType()->getPointerElementType(), ptr);
    auto align = module().getDataLayout().getABITypeAlign(ptr->getType()->getPointerElementType());
    result->setAlignment(align);
    return result;
}

llvm::Value* CodeGen::emit_store(llvm::IRBuilder<>& irbuilder, const Store* store) {
    emit_unsafe(store->mem());
    auto ptr = emit(store->ptr());
    auto result = irbuilder.CreateStore(emit(store->val()), ptr);
    auto align = module().getDataLayout().getABITypeAlign(ptr->getType()->getPointerElementType());
    result->setAlignment(align);
    return nullptr;
}

llvm::Value* CodeGen::emit_lea(llvm::IRBuilder<>& irbuilder, const LEA* lea) {
    if (lea->ptr_pointee()->isa<TupleType>() || lea->ptr_pointee()->isa<StructType>())
        return irbuilder.CreateStructGEP(convert(lea->ptr_pointee()), emit(lea->ptr()), primlit_value<u32>(lea->index()));

    assert(lea->ptr_pointee()->isa<ArrayType>() || lea->ptr_pointee()->isa<VectorType>());
    llvm::Value* args[2] = { irbuilder.getInt64(0), emit(lea->index()) };
    auto ptr = emit(lea->ptr());
    return irbuilder.CreateInBoundsGEP(ptr->getType()->getPointerElementType(), ptr, args);
}

llvm::Value* CodeGen::emit_assembly(llvm::IRBuilder<>& irbuilder, const Assembly* assembly) {
    emit_unsafe(assembly->mem());
    auto out_type = assembly->type()->isa<TupleType>();
    llvm::Type* res_type;
    bool mem_only = false;

    if (out_type) {
        if (out_type->num_ops() == 2)
            res_type = convert(out_type->types()[1]);
        else
            res_type = convert(world().tuple_type(out_type->types().skip_front()));
    } else {
        res_type = llvm::Type::getVoidTy(context());
        mem_only = true;
    }

    size_t num_inputs = assembly->num_inputs();
    auto input_values = Array<llvm::Value*>(num_inputs);
    auto input_types  = Array<llvm::Type* >(num_inputs);
    for (size_t i = 0; i != num_inputs; ++i) {
        input_values[i] = emit(assembly->input(i));
        input_types [i] = convert(assembly->input(i)->type());
    }

    auto fn_type = llvm::FunctionType::get(res_type, llvm_ref(input_types), false);

    std::string constraints;
    for (auto con : assembly->output_constraints())
        constraints += (constraints.empty() ? "" : ",") + con;
    for (auto con : assembly->input_constraints())
        constraints += (constraints.empty() ? "" : ",") + con;
    for (auto clob : assembly->clobbers())
        constraints += (constraints.empty() ? "" : ",") + std::string("~{") + clob + "}";
    // clang always marks those registers as clobbered, so we will do so as well
    if (llvm::Triple(module().getTargetTriple()).isX86())
        constraints += (constraints.empty() ? "" : ",") + std::string("~{dirflag},~{fpsr},~{flags}");

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

std::vector<llvm::Value*> CodeGen::emit_intrinsic(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    auto callee = body->callee()->as_nom<Continuation>();

    // Important: Must emit the memory object otherwise the memory
    // operations before the call to the intrinsic are all gone!
    for (size_t i = 0, e = body->num_args(); i != e; ++i) {
        if (is_mem(body->arg(i))) {
            emit_unsafe(body->arg(i));
            break;
        }
    }

    switch (callee->intrinsic()) {
        case Intrinsic::Atomic:       return { emit_atomic(irbuilder, continuation) };
        case Intrinsic::AtomicLoad:   return { emit_atomic_load(irbuilder, continuation) };
        case Intrinsic::AtomicStore:  emit_atomic_store(irbuilder, continuation); break;
        case Intrinsic::CmpXchg:      return emit_cmpxchg(irbuilder, continuation, false);
        case Intrinsic::CmpXchgWeak:  return emit_cmpxchg(irbuilder, continuation, true);
        case Intrinsic::Fence:        emit_fence(irbuilder, continuation); break;
        case Intrinsic::Reserve:      return { emit_reserve(irbuilder, continuation) };
        case Intrinsic::CUDA:         runtime_->emit_host_code(*this, irbuilder, Runtime::CUDA_PLATFORM,   ".cu",     continuation); break;
        case Intrinsic::NVVM:         runtime_->emit_host_code(*this, irbuilder, Runtime::CUDA_PLATFORM,   ".nvvm",   continuation); break;
        case Intrinsic::OpenCL:       runtime_->emit_host_code(*this, irbuilder, Runtime::OPENCL_PLATFORM, ".cl",     continuation); break;
        case Intrinsic::AMDGPU:       runtime_->emit_host_code(*this, irbuilder, Runtime::HSA_PLATFORM,    ".amdgpu", continuation); break;
        case Intrinsic::ShadyCompute: runtime_->emit_host_code(*this, irbuilder, Runtime::SHADY_PLATFORM,  ".shady",  continuation); break;
        case Intrinsic::HLS:          emit_hls(irbuilder, continuation);      break;
        case Intrinsic::Parallel:     emit_parallel(irbuilder, continuation); break;
        case Intrinsic::Fibers:       emit_fibers(irbuilder, continuation);   break;
        case Intrinsic::Spawn:        return { emit_spawn(irbuilder, continuation) };
        case Intrinsic::Sync:         emit_sync(irbuilder, continuation);     break;
#if THORIN_ENABLE_RV
        case Intrinsic::Vectorize:    emit_vectorize_continuation(irbuilder, continuation); break;
#else
        case Intrinsic::Vectorize:    throw std::runtime_error("rebuild with RV support");
#endif
        default: THORIN_UNREACHABLE;
    }

    return {};
}

llvm::Value* CodeGen::emit_atomic(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    assert(body->num_args() == 7 && "required arguments are missing");
    // atomic tag: Xchg Add Sub And Nand Or Xor Max Min UMax UMin FAdd FSub
    u32 binop_tag = body->arg(1)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicRMWInst::BinOp::Xchg) <= int(binop_tag) && int(binop_tag) <= int(llvm::AtomicRMWInst::BinOp::FSub) && "unsupported atomic");
    auto binop = (llvm::AtomicRMWInst::BinOp)binop_tag;
    auto is_valid_fop = is_type_f(body->arg(3)->type()) &&
                        (binop == llvm::AtomicRMWInst::BinOp::Xchg || binop == llvm::AtomicRMWInst::BinOp::FAdd || binop == llvm::AtomicRMWInst::BinOp::FSub);
    if (is_type_f(body->arg(3)->type()) && !is_valid_fop)
        world().edef(body->arg(3), "atomic {} is not supported for float types", binop_tag);
    else if (!is_type_i(body->arg(3)->type()) && !is_valid_fop)
        world().edef(body->arg(3), "atomic {} is only supported for int types", binop_tag);
    auto ptr = emit(body->arg(2));
    auto val = emit(body->arg(3));
    u32 order_tag = body->arg(4)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(order_tag) && int(order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)order_tag;
    auto scope = body->arg(5)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto call = irbuilder.CreateAtomicRMW(binop, ptr, val, llvm::MaybeAlign(), order, context().getOrInsertSyncScopeID(scope->as_string()));
    return call;
}

llvm::Value* CodeGen::emit_atomic_load(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    assert(body->num_args() == 5 && "required arguments are missing");
    auto ptr = emit(body->arg(1));
    u32 tag = body->arg(2)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(tag) && int(tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)tag;
    auto scope = body->arg(3)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto load = irbuilder.CreateLoad(ptr->getType()->getPointerElementType(), ptr);
    auto align = module().getDataLayout().getABITypeAlign(ptr->getType()->getPointerElementType());
    load->setAlignment(align);
    load->setAtomic(order, context().getOrInsertSyncScopeID(scope->as_string()));
    return load;
}

void CodeGen::emit_atomic_store(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    assert(body->num_args() == 6 && "required arguments are missing");
    auto ptr = emit(body->arg(1));
    auto val = emit(body->arg(2));
    u32 tag = body->arg(3)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(tag) && int(tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)tag;
    auto scope = body->arg(4)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto store = irbuilder.CreateStore(val, ptr);
    auto align = module().getDataLayout().getABITypeAlign(ptr->getType()->getPointerElementType());
    store->setAlignment(align);
    store->setAtomic(order, context().getOrInsertSyncScopeID(scope->as_string()));
}

std::vector<llvm::Value*> CodeGen::emit_cmpxchg(llvm::IRBuilder<>& irbuilder, Continuation* continuation, bool is_weak) {
    assert(continuation->has_body());
    auto body = continuation->body();

    assert(body->num_args() == 8 && "required arguments are missing");
    if (!is_type_i(body->arg(3)->type()))
        world().edef(body->arg(3), "cmpxchg only supported for integer types");
    auto ptr = emit(body->arg(1));
    auto cmp = emit(body->arg(2));
    auto val = emit(body->arg(3));
    u32 success_order_tag = body->arg(4)->as<PrimLit>()->qu32_value();
    u32 failure_order_tag = body->arg(5)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(success_order_tag) && int(success_order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(failure_order_tag) && int(failure_order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto success_order = (llvm::AtomicOrdering)success_order_tag;
    auto failure_order = (llvm::AtomicOrdering)failure_order_tag;
    auto scope = body->arg(6)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    auto call = irbuilder.CreateAtomicCmpXchg(ptr, cmp, val, llvm::MaybeAlign(), success_order, failure_order, context().getOrInsertSyncScopeID(scope->as_string()));
    call->setWeak(is_weak);
    return { irbuilder.CreateExtractValue(call, 0), irbuilder.CreateExtractValue(call, 1) };
}

void CodeGen::emit_fence(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    assert(body->num_args() == 4 && "required arguments are missing");
    u32 order_tag = body->arg(1)->as<PrimLit>()->qu32_value();
    assert(int(llvm::AtomicOrdering::NotAtomic) <= int(order_tag) && int(order_tag) <= int(llvm::AtomicOrdering::SequentiallyConsistent) && "unsupported atomic ordering");
    auto order = (llvm::AtomicOrdering)order_tag;
    auto scope = body->arg(2)->as<ConvOp>()->from()->as<Global>()->init()->as<DefiniteArray>();
    irbuilder.CreateFence(order, context().getOrInsertSyncScopeID(scope->as_string()));
}

llvm::Value* CodeGen::emit_reserve(llvm::IRBuilder<>&, const Continuation* continuation) {
    world().edef(continuation, "reserve_shared: only allowed in device code"); // TODO debug
    THORIN_UNREACHABLE;
}

llvm::Value* CodeGen::emit_reserve_shared(llvm::IRBuilder<>& irbuilder, const Continuation* continuation, bool init_undef) {
    assert(continuation->has_body());
    auto body = continuation->body();
    assert(body->num_args() == 3 && "required arguments are missing");
    if (!body->arg(1)->isa<PrimLit>())
        world().edef(body->arg(1), "reserve_shared: couldn't extract memory size");
    auto num_elems = body->arg(1)->as<PrimLit>()->ps32_value();
    auto cont_t = body->arg(2)->type()->as<FnType>();
    auto type = convert(cont_t->domain()[1]);
    // construct array type
    auto elem_type = cont_t->domain()[1]->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
    auto smem_type = this->convert(continuation->world().definite_array_type(elem_type, num_elems));
    auto name = continuation->unique_name();
    // NVVM doesn't allow '.' in global identifier
    std::replace(name.begin(), name.end(), '.', '_');
    auto global = emit_global_variable(smem_type, name, 3, init_undef);
    auto call = irbuilder.CreatePointerCast(global, type);
    return call;
}

/*
 * backend-specific stuff
 */

void CodeGen::emit_hls(llvm::IRBuilder<>& irbuilder, Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    std::vector<llvm::Value*> args(body->num_args()-3);
    for (size_t i = 2, j = 0; i < body->num_args(); ++i) {
        args[j++] = emit(body->arg(i));
    }
    auto callee = body->arg(1)->as_nom<Continuation>();
    world().make_external(callee);
    irbuilder.CreateCall(emit_fun_decl(callee), args);
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

}
