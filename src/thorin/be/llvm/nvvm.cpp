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

#include "thorin/world.h"

namespace thorin {

NVVMCodeGen::NVVMCodeGen(World& world, const Cont2Config& kernel_config)
    : CodeGen(world, llvm::CallingConv::C, llvm::CallingConv::PTX_Device, llvm::CallingConv::PTX_Kernel)
    , kernel_config_(kernel_config)
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
    llvm::Metadata* annotation_values_15[] = { llvm::ConstantAsMetadata::get(irbuilder_.getInt64(1)), llvm::ConstantAsMetadata::get(irbuilder_.getInt64(5)) };
    nvvmir_version_md->addOperand(llvm::MDNode::get(context_, annotation_values_15));
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

static u64 resolve_addr_space(const Def* def) {
    if (auto ptr = isa<Tag::Ptr>(def->type()))
        return as_lit<nat_t>(ptr->arg(1));
    return AddrSpace::Generic;
}

llvm::FunctionType* NVVMCodeGen::convert_fn_type(Lam* lam) {
    // skip non-global address-space parameters
    std::vector<const Def*> types;
    for (auto type : lam->type()->ops()) {
        if (auto ptr = isa<Tag::Ptr>(type))
            if (as_lit<nat_t>(ptr->arg(1)) == AddrSpace::Texture)
                continue;
        types.push_back(type);
    }
    return llvm::cast<llvm::FunctionType>(convert(lam->world().cn(types)));
}

void NVVMCodeGen::emit_function_decl_hook(Lam* lam, llvm::Function* f) {
    // append required metadata
    auto annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");

    const auto append_metadata = [&](llvm::Value* target, const std::string& name, const int val) {
        llvm::Metadata* annotation_values[] = { llvm::ValueAsMetadata::get(target), llvm::MDString::get(context_, name), llvm::ConstantAsMetadata::get(irbuilder_.getInt64(val)) };
        llvm::MDNode* result = llvm::MDNode::get(context_, annotation_values);
        annotation->addOperand(result);
        return result;
    };

    const auto emit_texture_kernel_arg = [&](const Def* var) {
        assert(as_lit<nat_t>(as<Tag::Ptr>(var->type())->arg(1)) == AddrSpace::Texture);
        auto global = emit_global_variable(irbuilder_.getInt64Ty(), var->debug().name, 1);
        metadata_[var] = append_metadata(global, "texture", 1);
    };

    append_metadata(f, "kernel", 1);

    auto config = kernel_config_.find(lam);
    if (config != kernel_config_.end()) {
        auto block = config->second->as<GPUKernelConfig>()->block_size();
        if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0) {
            append_metadata(f, "maxntidx", std::get<0>(block));
            append_metadata(f, "maxntidy", std::get<1>(block));
            append_metadata(f, "maxntidz", std::get<2>(block));
        }
    }

    // check signature for texturing memory
    for (auto var : lam->vars()) {
        if (auto ptr = isa<Tag::Ptr>(var->type())) {
            switch (as_lit<nat_t>(ptr->arg(1))) {
                case AddrSpace::Texture:
                    emit_texture_kernel_arg(var);
                    break;
                default:
                    // ignore this address space
                    break;
            }
        }
    }
}

llvm::Value* NVVMCodeGen::map_var(llvm::Function*, llvm::Argument* arg, const Def* var) {
    if (!get_var_lam(var)->is_external())
        return arg;
    else if (auto global = resolve_global_variable(var))
        return global;
    return arg;
}

llvm::Function* NVVMCodeGen::get_texture_handle_fun() {
    // %tex_ref = call i64 @llvm.nvvm.texsurf.handle.p1i64(metadata !{i64 addrspace(1)* @texture, metadata !"texture", i32 1}, i64 addrspace(1)* @texture)
    llvm::Type* types[2] = {
            llvm::Type::getMetadataTy(context_),
            llvm::PointerType::get(irbuilder_.getInt64Ty(), 1)
    };
    auto type = llvm::FunctionType::get(irbuilder_.getInt64Ty(), types, false);
    return llvm::cast<llvm::Function>(module_->getOrInsertFunction("llvm.nvvm.texsurf.handle.p1i64", type).getCallee()->stripPointerCasts());
}

void NVVMCodeGen::emit_function_start(llvm::BasicBlock*, Lam* lam) {
    if (!lam->is_external())
        return;
    // kernel needs special setup code for the arguments
    auto texture_handle = get_texture_handle_fun();
    for (auto var : lam->vars()) {
        if (auto global = resolve_global_variable(var)) {
            auto md = metadata_.find(var);
            assert(md != metadata_.end());
            // require specific handle to be mapped to a var
            llvm::Value* args[] = { llvm::MetadataAsValue::get(context_, md->second), global };
            vars_[var] = irbuilder_.CreateCall(texture_handle, args);
        }
    }
}

llvm::Value* NVVMCodeGen::emit_global(const Global* global) {
    world().wdef(global, "NVVM: Global variable '{}' will not be synced with host", global);
    return CodeGen::emit_global(global);
}

llvm::Value* NVVMCodeGen::emit_load(const App* load) {
    auto [mem, ptr] = load->args<2>();
    switch (resolve_addr_space(ptr)) {
        case AddrSpace::Texture:
            return irbuilder_.CreateExtractValue(lookup(ptr), { unsigned(0) });
        default:
            // shared address space uses the same load functionality
            return CodeGen::emit_load(load);
    }
}

llvm::Value* NVVMCodeGen::emit_store(const App* store) {
    auto [mem, ptr, val] = store->args<3>();
    assert(resolve_addr_space(ptr) != AddrSpace::Texture &&
            "Writes to textures are currently not supported");
    return CodeGen::emit_store(store);
}

static std::string get_texture_fetch_command(const Def* type) {
    std::stringstream fun_str;
    fun_str << "tex.1d.v4.";

     if (auto int_ = isa<Tag::Int>(type)) {
        switch (as_lit<u64>(int_->arg())) {
            case  1: fun_str << "s32"; break;
            case  8: fun_str << "s8";  break;
            case 16: fun_str << "s16"; break;
            case 32: fun_str << "s32"; break;
            case 64: fun_str << "s64"; break;
            default: THORIN_UNREACHABLE;
        }
    } else if (auto real = isa<Tag::Real>(type)) {
        switch (as_lit<u64>(real->arg())) {
            case 32: fun_str << "f32"; break;
            case 64: fun_str << "f64"; break;
            default: THORIN_UNREACHABLE;
        }
    }

    fun_str << ".s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];";
    return fun_str.str();
}

static std::string get_texture_fetch_constraint(const Def* type) {
    std::stringstream constraint_str;
    char c;

    if (auto int_ = isa<Tag::Int>(type)) {
        switch (as_lit<u64>(int_->arg())) {
            case  1: c = 'r'; break;
            case  8: c = 'c'; break;
            case 16: c = 'h'; break;
            case 32: c = 'r'; break;
            case 64: c = 'l'; break;
            default: THORIN_UNREACHABLE;
        }
    } else if (auto real = isa<Tag::Real>(type)) {
        switch (as_lit<u64>(real->arg())) {
            case 32: c = 'f'; break;
            case 64: c = 'd'; break;
            default: THORIN_UNREACHABLE;
        }
    }

    constraint_str << "=" << c << ",=" << c << ",=" << c << ",=" << c << ",l,r,r,r,r";
    return constraint_str.str();
}

llvm::Value* NVVMCodeGen::emit_lea(const App* lea) {
    auto [ptr, index] = lea->args<2>();
    switch (resolve_addr_space(ptr)) {
        case AddrSpace::Texture: {
            // sample for i32:
            // %tex_fetch = call { i32, i32, i32, i32 } asm sideeffect "tex.1d.v4.s32.s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];",
            // "=r,=r,=r,=r,l,r,r,r,r" (i64 %tex_ref, i32 %add, i32 0, i32 0, i32 0)
            auto ptr_ty = as<Tag::Ptr>(lea->type());
            auto [pointee, addr_space] = ptr_ty->args<2>();
            auto llvm_ptr_ty = convert(pointee);
            llvm::Type* struct_types[] = { llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty };
            auto ret_type = llvm::StructType::create(struct_types);
            llvm::Type* args[] = {
                irbuilder_.getInt64Ty(),
                irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty(), irbuilder_.getInt32Ty() };
            auto type = llvm::FunctionType::get(ret_type, args, false);
            auto fetch_command = get_texture_fetch_command(pointee);
            auto fetch_constraint = get_texture_fetch_constraint(pointee);
            auto get_call = llvm::InlineAsm::get(type, fetch_command, fetch_constraint, false);
            llvm::Value* values[] = {
                lookup(ptr), lookup(index),
                irbuilder_.getInt32(0), irbuilder_.getInt32(0), irbuilder_.getInt32(0) };
            return irbuilder_.CreateCall(get_call, values);
        }
        default:
            return CodeGen::emit_lea(lea);
    }
}

Lam* NVVMCodeGen::emit_reserve(Lam* lam) { return emit_reserve_shared(lam); }

llvm::GlobalVariable* NVVMCodeGen::resolve_global_variable(const Def* var) {
    if (resolve_addr_space(var) != AddrSpace::Global)
        return module_->getGlobalVariable(var->debug().name, true);
    return nullptr;
}

}
