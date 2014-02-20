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
    : CodeGen(world, llvm::CallingConv::C)
{}

void OpenCLCodeGen::emit() {
    std::ofstream file(world_.name() + ".cl");
    if (!file.is_open())
        throw std::runtime_error("cannot write '" + world_.name() + ".cl': " + strerror(errno));
    thorin::emit_c(world_, file, OPENCL);
    file.close();
}

Lambda* CodeGen::emit_opencl(Runtime& runtime, Lambda* lambda) {
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
    runtime.load_kernel(module_name, kernel_name);
    // fetch values and create external calls for initialization
    std::vector<std::pair<llvm::Value*, llvm::Constant*>> device_ptrs;
    for (size_t i = 5, e = lambda->num_args(); i < e; ++i) {
        Def opencl_param = lambda->arg(i);
        uint64_t num_elems = uint64_t(-1);
        if (const ArrayAgg* array_value = opencl_param->isa<ArrayAgg>())
            num_elems = (uint64_t)array_value->size();
        auto size = builder_.getInt64(num_elems);
        auto ptr = runtime.malloc(size);
//        device_ptrs.push_back(std::make_pair(loaded_device_ptr, size));
        runtime.write(ptr, lookup(opencl_param), size);
        runtime.set_kernel_arg(ptr);
    }
    // setup problem size
    runtime.set_problem_size(
        builder_.getInt64(it_space->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_space->op(2)->as<PrimLit>()->qu64_value()));
    // setup configuration
    runtime.set_config_size(
        builder_.getInt64(it_config->op(0)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(1)->as<PrimLit>()->qu64_value()),
        builder_.getInt64(it_config->op(2)->as<PrimLit>()->qu64_value()));
    // launch
    runtime.launch_kernel(kernel_name);
    // synchronize
    runtime.synchronize();

    // TODO back-fetch to CPU
    // TODO free mem
    return ret;
}

}
