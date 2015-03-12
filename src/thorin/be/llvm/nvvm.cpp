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

NVVMCodeGen::NVVMCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::PTX_Device, llvm::CallingConv::PTX_Kernel)
{
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    if (triple.isArch32Bit()) {
        module_->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module_->setTargetTriple("nvptx32-unknown-cuda");
    } else {
        module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module_->setTargetTriple("nvptx64-unknown-cuda");
    }
    // nvvmir.version
    auto nvvmir_version_md = module_->getOrInsertNamedMetadata("nvvmir.version");
    llvm::Value* annotation_values_11[] = { builder_.getInt32(1), builder_.getInt32(1) };
    nvvmir_version_md->addOperand(llvm::MDNode::get(context_, annotation_values_11));
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

static AddressSpace resolve_addr_space(Def def) {
    if (auto ptr = def->type().isa<PtrType>())
        return ptr->addr_space();
    return AddressSpace::Generic;
}

llvm::FunctionType* NVVMCodeGen::convert_fn_type(Lambda* lambda) {
    // skip non-global address-space parameters
    std::vector<Type> types;
    for (auto type : lambda->type()->args()) {
        if (auto ptr = type.isa<PtrType>())
            if (ptr->addr_space() == AddressSpace::Texture)
                continue;
        types.push_back(type);
    }
    return llvm::cast<llvm::FunctionType>(convert(lambda->world().fn_type(types)));
}

void NVVMCodeGen::emit_function_decl_hook(Lambda* lambda, llvm::Function* f) {
    // append required metadata
    auto annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");

    const auto append_metadata = [&](llvm::Value* target, const std::string &name) {
        llvm::Value* annotation_values[] = { target, llvm::MDString::get(context_, name), builder_.getInt64(1) };
        llvm::MDNode* result = llvm::MDNode::get(context_, annotation_values);
        annotation->addOperand(result);
        return result;
    };

    const auto emit_texture_kernel_arg = [&](const Param* param) {
        assert(param->type().as<PtrType>()->addr_space() == AddressSpace::Texture);
        auto global = emit_global_memory(builder_.getInt64Ty(), param->name, 1);
        metadata_[param] = append_metadata(global, "texture");
    };

    append_metadata(f, "kernel");

    // check signature for texturing memory
    for (auto param : lambda->params()) {
        if (auto ptr = param->type().isa<PtrType>()) {
            switch (ptr->addr_space()) {
                case AddressSpace::Texture:
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
    if (!param->lambda()->is_external())
        return arg;
    else if (auto var = resolve_global_variable(param))
        return var;
    return arg;
}

llvm::Function* NVVMCodeGen::get_texture_handle_fun() {
    // %tex_ref = call i64 @llvm.nvvm.texsurf.handle.p1i64(metadata !{i64 addrspace(1)* @texture, metadata !"texture", i32 1}, i64 addrspace(1)* @texture)
    llvm::Type* types[2] = {
            llvm::Type::getMetadataTy(context_),
            llvm::PointerType::get(builder_.getInt64Ty(), 1)
    };
    auto type = llvm::FunctionType::get(builder_.getInt64Ty(), types, false);
    return llvm::cast<llvm::Function>(module_->getOrInsertFunction("llvm.nvvm.texsurf.handle.p1i64", type));
}

void NVVMCodeGen::emit_function_start(llvm::BasicBlock* bb, Lambda* lambda) {
    if (!lambda->is_external())
        return;
    // kernel needs special setup code for the arguments
    auto texture_handle = get_texture_handle_fun();
    for (auto param : lambda->params()) {
        if (auto var = resolve_global_variable(param)) {
            auto md = metadata_.find(param);
            assert(md != metadata_.end());
            // require specific handle to be mapped to a parameter
            llvm::Value* args[] = { md->second, var };
            params_[param] = builder_.CreateCall(texture_handle, args);
        }
    }
}

llvm::Value* NVVMCodeGen::emit_load(Def def) {
    auto load = def->as<Load>();
    switch (resolve_addr_space(load->ptr())) {
        case AddressSpace::Texture:
            return builder_.CreateExtractValue(lookup(load->ptr()), { unsigned(0) });
        default:
            // shared address space uses the same load functionality
            return CodeGen::emit_load(def);
    }
}

llvm::Value* NVVMCodeGen::emit_store(Def def) {
    auto store = def->as<Store>();
    assert(resolve_addr_space(store->ptr()) != AddressSpace::Texture &&
            "Writes to textures are currently not supported");
    return CodeGen::emit_store(store);
}

static std::string get_texture_fetch_command(Type type) {
    std::stringstream fun_str;
    fun_str << "tex.1d.v4.";
    switch (type.as<PrimType>()->primtype_kind()) {
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

static std::string get_texture_fetch_constraint(Type type) {
    std::stringstream constraint_str;
    char c;
    switch (type.as<PrimType>()->primtype_kind()) {
        case PrimType_ps8:  case PrimType_qs8:
        case PrimType_pu8:  case PrimType_qu8:  c = 'c'; break; // not officially listed
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

llvm::Value* NVVMCodeGen::emit_lea(Def def) {
    auto lea = def->as<LEA>();
    switch (resolve_addr_space(lea->ptr())) {
        case AddressSpace::Texture: {
            // sample for i32:
            // %tex_fetch = call { i32, i32, i32, i32 } asm sideeffect "tex.1d.v4.s32.s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];",
            // "=r,=r,=r,=r,l,r,r,r,r" (i64 %tex_ref, i32 %add, i32 0, i32 0, i32 0)
            auto ptr_ty = lea->type();
            auto llvm_ptr_ty = convert(ptr_ty->referenced_type());
            llvm::Type* struct_types[] = { llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty };
            auto ret_type = llvm::StructType::create(struct_types);
            llvm::Type* args[] = {
                builder_.getInt64Ty(),
                builder_.getInt32Ty(), builder_.getInt32Ty(), builder_.getInt32Ty(), builder_.getInt32Ty() };
            auto type = llvm::FunctionType::get(ret_type, args, false);
            auto fetch_command = get_texture_fetch_command(ptr_ty->referenced_type());
            auto fetch_constraint = get_texture_fetch_constraint(ptr_ty->referenced_type());
            auto get_call = llvm::InlineAsm::get(type, fetch_command, fetch_constraint, false);
            llvm::Value* values[] = {
                lookup(lea->ptr()), lookup(lea->index()),
                builder_.getInt32(0), builder_.getInt32(0), builder_.getInt32(0) };
            return builder_.CreateCall(get_call, values);
        }
        default:
            return CodeGen::emit_lea(def);
    }
}

llvm::Value* NVVMCodeGen::emit_mmap(Def def) { return emit_shared_mmap(def); }

llvm::GlobalVariable* NVVMCodeGen::resolve_global_variable(const Param* param) {
    if (resolve_addr_space(param) != AddressSpace::Global)
        return module_->getGlobalVariable(param->name, true);
    return nullptr;
}

}
