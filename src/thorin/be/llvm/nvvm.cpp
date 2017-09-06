#include "thorin/be/llvm/nvvm.h"

#include <sstream>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

NVVMCodeGen::NVVMCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::PTX_Device, llvm::CallingConv::PTX_Kernel, kernel_config)
{
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    if (triple.isArch32Bit()) {
        module_->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module_->setTargetTriple("nvptx-nvidia-cuda");
    } else {
        module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module_->setTargetTriple("nvptx64-nvidia-cuda");
    }
    // nvvmir.version
    auto nvvmir_version_md = module_->getOrInsertNamedMetadata("nvvmir.version");
    llvm::Metadata* annotation_values_12[] = { llvm::ConstantAsMetadata::get(irbuilder_.getInt64(1)), llvm::ConstantAsMetadata::get(irbuilder_.getInt64(2)) };
    nvvmir_version_md->addOperand(llvm::MDNode::get(context_, annotation_values_12));
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
    for (auto type : continuation->type()->ops()) {
        if (auto ptr = type->isa<PtrType>())
            if (ptr->addr_space() == AddrSpace::Texture)
                continue;
        types.push_back(type);
    }
    return llvm::cast<llvm::FunctionType>(convert(continuation->world().fn_type(types)));
}

void NVVMCodeGen::emit_function_decl_hook(Continuation* continuation, llvm::Function* f) {
    // append required metadata
    auto annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");

    const auto append_metadata = [&](llvm::Value* target, const std::string& name) {
        llvm::Metadata* annotation_values[] = { llvm::ValueAsMetadata::get(target), llvm::MDString::get(context_, name), llvm::ConstantAsMetadata::get(irbuilder_.getInt64(1)) };
        llvm::MDNode* result = llvm::MDNode::get(context_, annotation_values);
        annotation->addOperand(result);
        return result;
    };

    const auto emit_texture_kernel_arg = [&](const Param* param) {
        assert(param->type()->as<PtrType>()->addr_space() == AddrSpace::Texture);
        auto global = emit_global_variable(irbuilder_.getInt64Ty(), param->name(), 1);
        metadata_[param] = append_metadata(global, "texture");
    };

    append_metadata(f, "kernel");

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
    if (!param->continuation()->is_external())
        return arg;
    else if (auto var = resolve_global_variable(param))
        return var;
    return arg;
}

llvm::Function* NVVMCodeGen::get_texture_handle_fun() {
    // %tex_ref = call i64 @llvm.nvvm.texsurf.handle.p1i64(metadata !{i64 addrspace(1)* @texture, metadata !"texture", i32 1}, i64 addrspace(1)* @texture)
    llvm::Type* types[2] = {
            llvm::Type::getMetadataTy(context_),
            llvm::PointerType::get(irbuilder_.getInt64Ty(), 1)
    };
    auto type = llvm::FunctionType::get(irbuilder_.getInt64Ty(), types, false);
    return llvm::cast<llvm::Function>(module_->getOrInsertFunction("llvm.nvvm.texsurf.handle.p1i64", type));
}

void NVVMCodeGen::emit_function_start(llvm::BasicBlock*, Continuation* continuation) {
    if (!continuation->is_external())
        return;
    // kernel needs special setup code for the arguments
    auto texture_handle = get_texture_handle_fun();
    for (auto param : continuation->params()) {
        if (auto var = resolve_global_variable(param)) {
            auto md = metadata_.find(param);
            assert(md != metadata_.end());
            // require specific handle to be mapped to a parameter
            llvm::Value* args[] = { llvm::MetadataAsValue::get(context_, md->second), var };
            params_[param] = irbuilder_.CreateCall(texture_handle, args);
        }
    }
}

llvm::Value* NVVMCodeGen::emit_global(const Global* global) {
    WLOG(global, "NVVM: Global variable '{}' will not be synced with host.", global);
    return CodeGen::emit_global(global);
}

llvm::Value* NVVMCodeGen::emit_load(const Load* load) {
    switch (resolve_addr_space(load->ptr())) {
        case AddrSpace::Texture:
            return irbuilder_.CreateExtractValue(lookup(load->ptr()), { unsigned(0) });
        default:
            // shared address space uses the same load functionality
            return CodeGen::emit_load(load);
    }
}

llvm::Value* NVVMCodeGen::emit_store(const Store* store) {
    assert(resolve_addr_space(store->ptr()) != AddrSpace::Texture &&
            "Writes to textures are currently not supported");
    return CodeGen::emit_store(store);
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

llvm::Value* NVVMCodeGen::emit_lea(const LEA* lea) {
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
                irbuilder_.getInt64Ty(),
                irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
            auto type = llvm::FunctionType::get(ret_type, args, false);
            auto fetch_command = get_texture_fetch_command(ptr_ty->pointee());
            auto fetch_constraint = get_texture_fetch_constraint(ptr_ty->pointee());
            auto get_call = llvm::InlineAsm::get(type, fetch_command, fetch_constraint, false);
            llvm::Value* values[] = {
                lookup(lea->ptr()), lookup(lea->index()),
                irbuilder_.getInt32(0), irbuilder_.getInt32(0), irbuilder_.getInt32(0) };
            return irbuilder_.CreateCall(get_call, values);
        }
        default:
            return CodeGen::emit_lea(lea);
    }
}

Continuation* NVVMCodeGen::emit_reserve(const Continuation* continuation) { return emit_reserve_shared(continuation); }

llvm::GlobalVariable* NVVMCodeGen::resolve_global_variable(const Param* param) {
    if (resolve_addr_space(param) != AddrSpace::Global)
        return module_->getGlobalVariable(param->name(), true);
    return nullptr;
}

}
