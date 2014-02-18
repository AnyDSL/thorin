#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/be/c.h"
#include "thorin/be/llvm/opencl.h"

#include <iostream>
#include <fstream>

namespace thorin {

OpenCLCodeGen::OpenCLCodeGen(World& world)
    : CodeGen(world, llvm::CallingConv::C) {}

void OpenCLCodeGen::emit() {
    std::ofstream file(world_.name() + ".cl");
    if (!file.is_open())
        throw std::runtime_error("cannot write '" + world_.name() + ".cl': " + strerror(errno));
    thorin::emit_c(world_, file, OPENCL);
    file.close();
}

Lambda* CodeGen::emit_opencl(Lambda* lambda) {
    // to-target is the desired OpenCL call
    // target(mem, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = lambda->to()->as_lambda();
    assert(target->is_builtin() && target->attribute().is(Lambda::OPENCL));
    assert(lambda->num_args() > 4 && "required arguments are missing");

    // get input
    auto it_space  = lambda->arg(1)->as<Tuple>();
    auto it_config = lambda->arg(2)->as<Tuple>();
    auto kernel = lambda->arg(3)->as<Global>()->init()->as<Lambda>()->name;
    auto ret = lambda->arg(4)->as_lambda();

    // load kernel
    auto module_name = builder_.CreateGlobalStringPtr(world_.name() + ".cl");
    auto kernel_name = builder_.CreateGlobalStringPtr(kernel);
    llvm::Value* load_args[] = { module_name, kernel_name };
    builder_.CreateCall(opencl("spir_build_program_and_kernel_from_source"), load_args);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def opencl_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = opencl_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto alloca = builder_.CreateAlloca(opencl_device_ptr_ty_);
        auto device_ptr = builder_.CreateCall(opencl("spir_malloc_buffer"), size);
        // store device ptr
        builder_.CreateStore(device_ptr, alloca);
        auto loaded_device_ptr = builder_.CreateLoad(alloca);
        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        llvm::Value* mem_args[] = {
            loaded_device_ptr,
            builder_.CreateBitCast(lookup(opencl_param), llvm::Type::getInt8PtrTy(context_)),
            size
        };
        builder_.CreateCall(opencl("spir_write_buffer"), mem_args);
        // set_kernel_arg(void *, size_t)
        auto *DL = new llvm::DataLayout(module_.get());
        auto size_of_arg = builder_.getInt64(DL->getTypeAllocSize(llvm::Type::getInt8PtrTy(context_)));
        llvm::Value* arg_args[] = { alloca, size_of_arg };
        builder_.CreateCall(opencl("spir_set_kernel_arg"), arg_args);
    }
    // setup problem size
    llvm::Value* problem_size_args[] = {
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(opencl("spir_set_problem_size"), problem_size_args);
    // setup configuration
    llvm::Value* config_args[] = {
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value())
    };
    builder_.CreateCall(opencl("spir_set_config_size"), config_args);
    // launch
    builder_.CreateCall(opencl("spir_launch_kernel"), { kernel_name });
    // synchronize
    builder_.CreateCall(opencl("spir_synchronize"));

    // back-fetch to CPU
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def opencl_param = lambda->arg(i);
        auto entry = device_ptrs[i - 5];
        // need to fetch back memory
        llvm::Value* args[] = {
            entry.first,
            builder_.CreateBitCast(lookup(opencl_param), llvm::Type::getInt8PtrTy(context_)),
            entry.second
        };
        builder_.CreateCall(opencl("spir_read_buffer"), args);
    }

    // free memory
    for (auto device_ptr : device_ptrs)
        builder_.CreateCall(opencl("spir_free_buffer"), { device_ptr.first });
    return ret;
}

}
