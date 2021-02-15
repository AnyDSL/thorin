#include "backends.h"

#include "thorin/analyses/scope.h"

#ifdef THORIN_ENABLE_LLVM
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu.h"
#include "thorin/be/llvm/cuda.h"
#include "thorin/be/llvm/hls.h"
#include "thorin/be/c/opencl.h"
#include "thorin/transform/codegen_prepare.h"
#endif

namespace thorin {

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

#ifdef THORIN_ENABLE_LLVM
    cpu_cg = std::make_unique<llvm_be::CPUCodeGen>(world, opt, debug);

    if (!nvvm.  world().empty()) nvvm_cg   = std::make_unique<llvm_be::NVVMCodeGen  >(nvvm  .world(), kernel_config,      debug);
    if (!amdgpu.world().empty()) amdgpu_cg = std::make_unique<llvm_be::AMDGPUCodeGen>(amdgpu.world(), kernel_config, opt, debug);
#else
    // TODO: maybe use the C backend as a fallback when LLVM is not present for host codegen ?
#endif
    // TODO: Fix the C-based backends
    //if (!cuda.  world().empty()) cuda_cg   = std::make_unique<CUDACodeGen  >(cuda  .world(), kernel_config, opt, debug);
    //if (!opencl.world().empty()) opencl_cg = std::make_unique<OpenCLCodeGen>(opencl.world(), kernel_config, opt, debug);
    //if (!hls.   world().empty()) hls_cg    = std::make_unique<HLSCodeGen   >(hls   .world(), kernel_config, opt, debug);
}

CodeGen::CodeGen(World& world, bool debug)
: world_(world)
, debug_(debug)
{}

}
