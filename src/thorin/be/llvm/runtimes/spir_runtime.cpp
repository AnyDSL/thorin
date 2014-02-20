#include "thorin/be/llvm/runtimes/spir_runtime.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

SpirRuntime::SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder)
    : Runtime(context, target, builder, llvm::IntegerType::getInt64Ty(context), "spir.s")
{
    auto *DL = new llvm::DataLayout(target);
    size_of_kernel_arg_ = builder_.getInt64(DL->getTypeAllocSize(llvm::Type::getInt8PtrTy(context)));
}

llvm::Value* SpirRuntime::malloc(llvm::Value* size) {
    auto alloca = builder_.CreateAlloca(get_device_ptr_ty());
    auto device_ptr = builder_.CreateCall(get("spir_malloc_buffer"), size);
    builder_.CreateStore(device_ptr, alloca);
    return alloca;
}

llvm::CallInst* SpirRuntime::free(llvm::Value* ptr) {
    return builder_.CreateCall(get("spir_free_buffer"), { ptr });
}

llvm::CallInst* SpirRuntime::write(llvm::Value* ptr, llvm::Value* data, llvm::Value* size) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()), size };
    return builder_.CreateCall(get("spir_write_buffer"), mem_args);
}

llvm::CallInst* SpirRuntime::read(llvm::Value* ptr, llvm::Value* data, llvm::Value* length) {
    llvm::Value* args[] = { ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()), length };
    return builder_.CreateCall(get("spir_read_buffer"), args);
}

llvm::CallInst* SpirRuntime::set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { x, y, z };
    return builder_.CreateCall(get("spir_set_problem_size"), problem_size_args);
}

llvm::CallInst* SpirRuntime::set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { x, y, z };
    return builder_.CreateCall(get("spir_set_config_size"), config_args);
}

llvm::CallInst* SpirRuntime::synchronize() {
    return builder_.CreateCall(get("spir_synchronize"));
}

llvm::CallInst* SpirRuntime::set_kernel_arg(llvm::Value* ptr) {
    llvm::Value* arg_args[] = { ptr, size_of_kernel_arg_ };
    return builder_.CreateCall(get("spir_set_kernel_arg"), arg_args);
}

llvm::CallInst* SpirRuntime::load_kernel(llvm::Value* module, llvm::Value* data) {
    llvm::Value* load_args[] = { module, data };
    return builder_.CreateCall(get("spir_build_program_and_kernel_from_binary"), load_args);
}

llvm::CallInst* SpirRuntime::launch_kernel(llvm::Value* name) {
    return builder_.CreateCall(get("spir_launch_kernel"), { name });
}

Lambda* SpirRuntime::emit_host_code(CodeGen& code_gen, Lambda* lambda) {
    auto &world = lambda->world();
    // to-target is the desired SPIR call
    // target(mem, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && (target->attribute().is(Lambda::SPIR) || target->attribute().is(Lambda::OPENCL)));
    assert(lambda->num_args() > 4 && "required arguments are missing");

    // get input
    auto it_space  = lambda->arg(1)->as<Tuple>();
    auto it_config = lambda->arg(2)->as<Tuple>();
    auto kernel = lambda->arg(3)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(4)->as_lambda();

    // load kernel
    auto module_name = builder_.CreateGlobalStringPtr(world.name() + "_spir.bc");
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    load_kernel(module_name, kernel_name);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def spir_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = spir_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto ptr = malloc(size);
//        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        write(ptr, code_gen.lookup(spir_param), size);
        set_kernel_arg(ptr);
    }
    // setup problem size
    set_problem_size(
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value()));
    // setup configuration
    set_config_size(
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value()));
    // launch
    launch_kernel(kernel_name);
    // synchronize
    synchronize();

    // TODO back-fetch to CPU
    // TODO free mem
    return ret;
}

}
