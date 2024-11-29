#include "thorin/be/llvm/nvvm.h"

#include <sstream>
#include <unordered_map> // TODO don't used std::unordered_*

#include <llvm/TargetParser/Triple.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin::llvm {

NVVMCodeGen::NVVMCodeGen(Thorin& thorin, const Cont2Config& kernel_config, int /* opt */, bool /* debug */)
    : CodeGen(thorin, llvm::CallingConv::C, llvm::CallingConv::PTX_Device, llvm::CallingConv::PTX_Kernel, 0, false)
    , kernel_config_(kernel_config)
{
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    if (triple.isArch32Bit()) {
        module().setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module().setTargetTriple("nvptx-nvidia-cuda");
    } else {
        module().setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module().setTargetTriple("nvptx64-nvidia-cuda");
    }
    // nvvmir.version
    auto nvvmir_version_md = module().getOrInsertNamedMetadata("nvvmir.version");
    llvm::IRBuilder<> irbuilder(context());
    llvm::Metadata* annotation_values_15[] = {
        llvm::ConstantAsMetadata::get(irbuilder.getInt64(1)),
        llvm::ConstantAsMetadata::get(irbuilder.getInt64(5))
    };
    nvvmir_version_md->addOperand(llvm::MDNode::get(context(), annotation_values_15));
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

static AddrSpace resolve_addr_space(const Def* def) {
    if (auto ptr = def->type()->isa<PtrType>())
        return ptr->addr_space();
    return AddrSpace::Generic;
}

llvm::FunctionType* NVVMCodeGen::convert_fn_type(Continuation* continuation) {
    // skip non-global address-space parameters
    std::vector<const Type*> types;
    for (auto type : continuation->type()->types()) {
        if (auto ptr = type->isa<PtrType>())
            if (ptr->addr_space() == AddrSpace::Texture)
                continue;
        types.push_back(type);
    }
    return llvm::cast<llvm::FunctionType>(convert(continuation->world().fn_type(types)));
}

void NVVMCodeGen::emit_fun_decl_hook(Continuation* continuation, llvm::Function* f) {
    auto annotation = module().getOrInsertNamedMetadata("nvvm.annotations");
    auto int64_type = llvm::IntegerType::get(context(), 64);

    const auto append_metadata = [&](llvm::Value* target, const std::string& name, const int val) {
        llvm::Metadata* annotation_values[] = {
            llvm::ValueAsMetadata::get(target),
            llvm::MDString::get(context(), name),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int64_type, val, true))
        };
        llvm::MDNode* result = llvm::MDNode::get(context(), annotation_values);
        annotation->addOperand(result);
        return result;
    };

    const auto emit_texture_kernel_arg = [&](const Param* param) {
        assert(param->type()->as<PtrType>()->addr_space() == AddrSpace::Texture);
        auto global = emit_global_variable(int64_type, param->name(), 1);
        metadata_[param] = append_metadata(global, "texture", 1);
    };

    append_metadata(f, "kernel", 1);

    auto config = kernel_config_.find(continuation);
    if (config != kernel_config_.end()) {
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            append_metadata(f, "maxntidx", std::get<0>(block));
            append_metadata(f, "maxntidy", std::get<1>(block));
            append_metadata(f, "maxntidz", std::get<2>(block));
        }
    }

    // check signature for texturing memory
    for (auto param : continuation->params()) {
        if (auto ptr = param->type()->isa<PtrType>()) {
            switch (ptr->addr_space()) {
                case AddrSpace::Texture:
                    emit_texture_kernel_arg(param);
                    break;
                default:
                    // ignore this address space
                    break;
            }
        }
    }
}

llvm::Value* NVVMCodeGen::map_param(llvm::Function*, llvm::Argument* arg, const Param* param) {
    if (!param->continuation()->is_exported())
        return arg;
    else if (auto var = resolve_global_variable(param))
        return var;
    return arg;
}

llvm::Function* NVVMCodeGen::get_texture_handle_fun(llvm::IRBuilder<>& irbuilder) {
    // %tex_ref = call i64 @llvm.nvvm.texsurf.handle.p1i64(metadata !{i64 addrspace(1)* @texture, metadata !"texture", i32 1}, i64 addrspace(1)* @texture)
    llvm::Type* types[2] = {
        llvm::Type::getMetadataTy(context()),
        llvm::PointerType::get(irbuilder.getInt64Ty(), 1)
    };
    auto type = llvm::FunctionType::get(irbuilder.getInt64Ty(), types, false);
    return llvm::cast<llvm::Function>(module().getOrInsertFunction("llvm.nvvm.texsurf.handle.p1i64", type).getCallee()->stripPointerCasts());
}

void NVVMCodeGen::prepare(Continuation* cont, llvm::Function* fct) {
    CodeGen::prepare(cont, fct);

    if (cont != entry_ || !cont->is_exported()) return;

    auto& irbuilder = *cont2bb_[cont].second;
    // kernel needs special setup code for the arguments
    auto texture_handle = get_texture_handle_fun(irbuilder);
    for (auto param : cont->params()) {
        if (auto var = resolve_global_variable(param)) {
            auto md = metadata_.find(param);
            assert(md != metadata_.end());
            // require specific handle to be mapped to a parameter
            llvm::Value* args[] = { llvm::MetadataAsValue::get(context(), md->second), var };
            defs_[param] = irbuilder.CreateCall(texture_handle, args);
        }
    }
}

llvm::Value* NVVMCodeGen::emit_global(const Global* global) {
    if (global->is_mutable())
        world().wdef(global, "NVVM: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

llvm::Value* NVVMCodeGen::emit_load(llvm::IRBuilder<>& irbuilder, const Load* load) {
    switch (resolve_addr_space(load->ptr())) {
        case AddrSpace::Texture:
            return irbuilder.CreateExtractValue(emit(load->ptr()), { unsigned(0) });
        default:
            // shared address space uses the same load functionality
            return CodeGen::emit_load(irbuilder, load);
    }
}

llvm::Value* NVVMCodeGen::emit_store(llvm::IRBuilder<>& irbuilder, const Store* store) {
    assert(resolve_addr_space(store->ptr()) != AddrSpace::Texture &&
            "Writes to textures are currently not supported");
    return CodeGen::emit_store(irbuilder, store);
}

static std::string get_texture_fetch_command(const Type* type) {
    std::stringstream fun_str;
    fun_str << "tex.1d.v4.";
    switch (type->as<PrimType>()->primtype_tag()) {
        case PrimType_ps8:  case PrimType_qs8:
        case PrimType_pu8:  case PrimType_qu8:  fun_str << "s8";  break;
        case PrimType_ps16: case PrimType_qs16:
        case PrimType_pu16: case PrimType_qu16: fun_str << "s16"; break;
        case PrimType_bool:
        case PrimType_ps32: case PrimType_qs32:
        case PrimType_pu32: case PrimType_qu32: fun_str << "s32"; break;
        case PrimType_ps64: case PrimType_qs64:
        case PrimType_pu64: case PrimType_qu64: fun_str << "s64"; break;
        case PrimType_pf32: case PrimType_qf32: fun_str << "f32"; break;
        case PrimType_pf64: case PrimType_qf64: fun_str << "f64"; break;
        default:
            THORIN_UNREACHABLE;
    }
    fun_str << ".s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];";
    return fun_str.str();
}

static std::string get_texture_fetch_constraint(const Type* type) {
    std::stringstream constraint_str;
    char c;
    switch (type->as<PrimType>()->primtype_tag()) {
        case PrimType_ps8:  case PrimType_qs8:
        case PrimType_pu8:  case PrimType_qu8:  c = 'c'; break;
        case PrimType_ps16: case PrimType_qs16:
        case PrimType_pu16: case PrimType_qu16: c = 'h'; break;
        case PrimType_bool:
        case PrimType_ps32: case PrimType_qs32:
        case PrimType_pu32: case PrimType_qu32: c = 'r'; break;
        case PrimType_ps64: case PrimType_qs64:
        case PrimType_pu64: case PrimType_qu64: c = 'l'; break;
        case PrimType_pf32: case PrimType_qf32: c = 'f'; break;
        case PrimType_pf64: case PrimType_qf64: c = 'd'; break;
        default:
            THORIN_UNREACHABLE;
    }
    constraint_str << "=" << c << ",=" << c << ",=" << c << ",=" << c
                   << ",l,r,r,r,r";
    return constraint_str.str();
}

llvm::Value* NVVMCodeGen::emit_lea(llvm::IRBuilder<>& irbuilder, const LEA* lea) {
    switch (resolve_addr_space(lea->ptr())) {
        case AddrSpace::Texture: {
            // sample for i32:
            // %tex_fetch = call { i32, i32, i32, i32 } asm sideeffect "tex.1d.v4.s32.s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];",
            // "=r,=r,=r,=r,l,r,r,r,r" (i64 %tex_ref, i32 %add, i32 0, i32 0, i32 0)
            auto ptr_ty = lea->type();
            auto llvm_ptr_ty = convert(ptr_ty->pointee());
            llvm::Type* struct_types[] = { llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty };
            auto ret_type = llvm::StructType::create(struct_types);
            llvm::Type* args[] = {
                irbuilder.getInt64Ty(),
                irbuilder.getInt32Ty(), irbuilder.getInt32Ty(), irbuilder.getInt32Ty(), irbuilder.getInt32Ty() };
            auto type = llvm::FunctionType::get(ret_type, args, false);
            auto fetch_command = get_texture_fetch_command(ptr_ty->pointee());
            auto fetch_constraint = get_texture_fetch_constraint(ptr_ty->pointee());
            auto get_call = llvm::InlineAsm::get(type, fetch_command, fetch_constraint, false);
            llvm::Value* values[] = {
                emit(lea->ptr()), emit(lea->index()),
                irbuilder.getInt32(0), irbuilder.getInt32(0), irbuilder.getInt32(0) };
            return irbuilder.CreateCall(get_call, values);
        }
        default:
            return CodeGen::emit_lea(irbuilder, lea);
    }
}

llvm::Value* NVVMCodeGen::emit_reserve(llvm::IRBuilder<>& irbuilder, const Continuation* continuation) {
    return emit_reserve_shared(irbuilder, continuation);
}

llvm::Value* NVVMCodeGen::emit_mathop(llvm::IRBuilder<>& irbuilder, const MathOp* mathop) {
    auto make_key = [] (MathOpTag tag, unsigned bitwidth) { return (static_cast<unsigned>(tag) << 16) | bitwidth; };
    static const std::unordered_map<unsigned, std::string> libdevice_functions = {
#define MATH_FUNCTION(name) \
        { make_key(MathOp_##name, 32), "__nv_" #name "f" }, \
        { make_key(MathOp_##name, 64), "__nv_" #name },
        MATH_FUNCTION(fabs)
        MATH_FUNCTION(copysign)
        MATH_FUNCTION(round)
        MATH_FUNCTION(floor)
        MATH_FUNCTION(ceil)
        MATH_FUNCTION(fmin)
        MATH_FUNCTION(fmax)
        MATH_FUNCTION(cos)
        MATH_FUNCTION(sin)
        MATH_FUNCTION(tan)
        MATH_FUNCTION(acos)
        MATH_FUNCTION(asin)
        MATH_FUNCTION(atan)
        MATH_FUNCTION(atan2)
        MATH_FUNCTION(sqrt)
        MATH_FUNCTION(cbrt)
        MATH_FUNCTION(pow)
        MATH_FUNCTION(exp)
        MATH_FUNCTION(exp2)
        MATH_FUNCTION(log)
        MATH_FUNCTION(log2)
        MATH_FUNCTION(log10)
#undef MATH_FUNCTION
    };
    auto key = make_key(mathop->mathop_tag(), num_bits(mathop->type()->primtype_tag()));
    auto call = call_math_function(irbuilder, mathop, libdevice_functions.at(key));
    llvm::cast<llvm::CallInst>(call)->setCallingConv(function_calling_convention_);
    return call;
}

llvm::GlobalVariable* NVVMCodeGen::resolve_global_variable(const Param* param) {
    if (resolve_addr_space(param) != AddrSpace::Global)
        return module().getGlobalVariable(param->name(), true);
    return nullptr;
}

}
