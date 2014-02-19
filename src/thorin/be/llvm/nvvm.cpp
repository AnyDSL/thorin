#include "thorin/be/llvm/nvvm.h"
#include <sstream>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/memop.h"

namespace thorin {

NVVMCodeGen::NVVMCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::PTX_Device)
{
    module_->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
}

//------------------------------------------------------------------------------
// Kernel code
//------------------------------------------------------------------------------

static AddressSpace resolve_addr_space(Def def) {
    if (auto ptr = def->type()->isa<Ptr>())
        return ptr->addr_space();
    return AddressSpace::Global;
}

llvm::Function* NVVMCodeGen::emit_function_decl(std::string& name, Lambda* lambda) {
    // skip non-global address-space parameters
    std::vector<const Type*> types;
    for (auto type : lambda->type()->elems()) {
        if (auto ptr = type->isa<Ptr>())
            if (ptr->addr_space() != AddressSpace::Global)
                continue;
        types.push_back(type);
    }
    auto ft = llvm::cast<llvm::FunctionType>(map(lambda->world().pi(types)));
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, lambda->name, module_);

    if (!lambda->attribute().is(Lambda::KernelEntry))
        return f;

    // append required metadata
    auto annotation = module_->getOrInsertNamedMetadata("nvvm.annotations");

    const auto append_metadata = [&](llvm::Value* target, const std::string &name) {
        llvm::Value* annotation_values[] = { target, llvm::MDString::get(context_, name), builder_.getInt64(1) };
        llvm::MDNode* result = llvm::MDNode::get(context_, annotation_values);
        annotation->addOperand(result);
        return result;
    };

    const auto emit_texture_kernel_arg = [&](const Param* param) {
        assert(param->type()->as<Ptr>()->addr_space() == AddressSpace::Texture);
        auto global = new llvm::GlobalVariable(*module_.get(), builder_.getInt64Ty(), false,
                llvm::GlobalValue::InternalLinkage, builder_.getInt64(0), param->unique_name(),
                nullptr, llvm::GlobalVariable::NotThreadLocal, 1);
        metadata_[param] = append_metadata(global, "texture");
    };

    append_metadata(f, "kernel");
    f->setCallingConv(llvm::CallingConv::PTX_Kernel);

    // check signature for texturing memory
    for (auto param : lambda->params()) {
        if (auto ptr = param->type()->isa<Ptr>()){
            switch (ptr->addr_space()) {
            case AddressSpace::Texture:
                emit_texture_kernel_arg(param);
                break;
            case AddressSpace::Shared:
                assert(false && "Shared address space is TODO");
                break;
            default:
                // ignore this address space
                break;
            }
        }
    }

    return f;
}

llvm::Value* NVVMCodeGen::map_param(llvm::Function*, llvm::Argument* arg, const Param* param) {
    if (!param->lambda()->attribute().is(Lambda::KernelEntry))
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

void NVVMCodeGen::emit_function_start(llvm::BasicBlock* bb, llvm::Function* f, Lambda* lambda) {
    if (!lambda->attribute().is(Lambda::KernelEntry))
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

llvm::Function* NVVMCodeGen::emit_intrinsic_decl(std::string& name, Lambda* lambda) {
    auto f = CodeGen::emit_function_decl(name, lambda);
    f->setAttributes(llvm::AttributeSet());
    return f;
}

llvm::Value* NVVMCodeGen::emit_load(Def def) {
    auto load = def->as<Load>();
    switch (resolve_addr_space(load->ptr())) {
    case AddressSpace::Texture:
        return builder_.CreateExtractValue(lookup(load->ptr()), { unsigned(0) });
    case AddressSpace::Shared:
        THORIN_UNREACHABLE;
    default:
        return CodeGen::emit_load(def);
    }
}

llvm::Value* NVVMCodeGen::emit_store(Def def) {
    auto store = def->as<Store>();
    assert(resolve_addr_space(store->ptr()) == AddressSpace::Global &&
            "Only global address space for stores is currently supported");
    return CodeGen::emit_store(store);
}

static std::string get_texture_fetch_command(const Type* type) {
    std::stringstream fun_str;
    if (type->as<PrimType>()->is_type_f())
        fun_str << "tex.1d.v4.f32.s32";
    else
        fun_str << "tex.1d.v4.s32.s32";
    fun_str << " {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];";
    return fun_str.str();
}

llvm::Value* NVVMCodeGen::emit_lea(Def def) {
    auto lea = def->as<LEA>();
    switch (resolve_addr_space(lea->ptr())) {
    case AddressSpace::Texture: {
        // sample for i32:
        // %tex_fetch = call { i32, i32, i32, i32 } asm sideeffect "tex.1d.v4.s32.s32 {$0,$1,$2,$3}, [$4, {$5,$6,$7,$8}];",
        // "=r,=r,=r,=r,l,r,r,r,r" (i64 %tex_ref, i32 %add, i32 0, i32 0, i32 0)
        auto ptr_ty = lea->type()->as<Ptr>();
        auto llvm_ptr_ty = map(ptr_ty->referenced_type());
        llvm::Type* struct_types[] = { llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty };
        auto ret_type = llvm::StructType::create(struct_types);
        llvm::Type* args[] = {
            builder_.getInt64Ty(),
            builder_.getInt32Ty(), builder_.getInt32Ty(), builder_.getInt32Ty(), builder_.getInt32Ty() };
        auto type = llvm::FunctionType::get(ret_type, args, false);
        auto fetch_command = get_texture_fetch_command(ptr_ty->referenced_type());
        auto get_call = llvm::InlineAsm::get(type, fetch_command, "=r,=r,=r,=r,l,r,r,r,r", false);
        llvm::Value* values[] = {
            lookup(lea->ptr()), lookup(lea->index()),
            builder_.getInt32(0), builder_.getInt32(0), builder_.getInt32(0) };
        return builder_.CreateCall(get_call, values);
    }
    default:
        return CodeGen::emit_lea(def);
    }
}

llvm::GlobalVariable* NVVMCodeGen::resolve_global_variable(const Param* param) {
    if (resolve_addr_space(param) != AddressSpace::Global)
        return module_->getGlobalVariable(param->unique_name(), true);
    return nullptr;
}

//------------------------------------------------------------------------------
// Host code
//------------------------------------------------------------------------------

Lambda* CodeGen::emit_nvvm(Lambda* lambda) {
    // to-target is the desired CUDA call
    // target(mem, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::NVVM));
    assert(lambda->num_args() > 4 && "required arguments are missing");

    // get input
    auto it_space  = lambda->arg(1)->as<Tuple>();
    auto it_config = lambda->arg(2)->as<Tuple>();
    auto kernel = lambda->arg(3)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(4)->as_lambda();

    // load kernel
    auto module_name = builder_.CreateGlobalStringPtr(world_.name() + "_nvvm.ll");
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(nvvm("nvvm_load_kernel"), load_args);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = cuda_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto alloca = builder_.CreateAlloca(nvvm_device_ptr_ty_);
        auto device_ptr = builder_.CreateCall(nvvm("nvvm_malloc_memory"), size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), size };
        builder_.CreateCall(nvvm("nvvm_write_memory"), mem_args);
        builder_.CreateCall(nvvm("nvvm_set_kernel_arg"), { alloca });
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(nvvm("nvvm_set_problem_size"), problem_size_args);
    // setup configuration
    llvm::Value* config_args[] = {
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(nvvm("nvvm_set_config_size"), config_args);
    // launch
    builder_.CreateCall(nvvm("nvvm_launch_kernel"), { kernel_name });
    // synchronize
    builder_.CreateCall(nvvm("nvvm_synchronize"));

    // back-fetch to CPU
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def cuda_param = lambda->arg(i);
        auto entry = device_ptrs[i - 5];
        // need to fetch back memory
        llvm::Value* args[] = { entry.first, builder_.CreateBitCast(lookup(cuda_param), builder_.getInt8PtrTy()), entry.second };
        builder_.CreateCall(nvvm("nvvm_read_memory"), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(nvvm("nvvm_free_memory"), { device_ptr.first });
    return ret;
}

}
