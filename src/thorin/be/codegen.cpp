#include "thorin/be/codegen.h"

#include "thorin/be/c/c.h"
#include "thorin/be/runtime.h"

#if THORIN_ENABLE_LLVM
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu_hsa.h"
#include "thorin/be/llvm/amdgpu_pal.h"
#endif

#if THORIN_ENABLE_SHADY
#include "thorin/be/shady/shady.h"
#undef empty
#undef nodes
#endif

#if THORIN_ENABLE_SPIRV
#include "thorin/be/spirv/spirv.h"
#endif

#include "thorin/transform/hls_channels.h"
#include "thorin/transform/hls_kernel_launch.h"
#include "lower_offload_intrinsics.h"

namespace thorin {

void Backend::prepare_kernel_configs() {
    device_code_.opt();

    Cont2Config adjusted_configs_map;

    auto conts = device_code_.world().copy_continuations();
    for (auto& [continuation, config] : kernel_configs_) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto original_cont : conts) {
            if (!original_cont) continue;
            if (!original_cont->has_body()) continue;
            if (original_cont->name() == continuation->name())
                imported = original_cont;
        }
        assert(imported && "we lost a kernel ?");
        if (!imported) continue;

        adjusted_configs_map[imported] = std::move(config);
    }

    std::swap(kernel_configs_, adjusted_configs_map);
}

static const App* get_alloc_call(const Def* def) {
    // look through casts
    while (auto conv_op = def->isa<ConvOp>())
        def = conv_op->op(0);

    auto param = def->isa<Param>();
    if (!param) return nullptr;

    auto ret = param->continuation();
    for (auto use : ret->uses()) {
        auto call = use.def()->isa<App>();
        if (!call || use.index() == 0) continue;

        auto callee = call->callee();
        if (callee->name() != "anydsl_alloc") continue;

        return call;
    }
    return nullptr;
}

static uint64_t get_alloc_size(const Def* def) {
    auto call = get_alloc_call(def);
    if (!call) return 0;

    // signature: anydsl_alloc(mem, i32, i64, fn(mem, &[i8]))
    auto size = call->arg(2)->isa<PrimLit>();
    return size ? static_cast<uint64_t>(size->value().get_qu64()) : 0_u64;
}

static std::unique_ptr<GPUKernelConfig> get_gpu_kernel_config(const App* app, Continuation* /* imported */) {
    // determine whether or not this kernel uses restrict pointers
    bool has_restrict = true;
    DefSet allocs;
    for (size_t i = KernelLaunchArgs::Num, e = app->num_args(); has_restrict && i != e; ++i) {
        auto arg = app->arg(i);
        if (!arg->type()->isa<PtrType>()) continue;
        auto alloc = get_alloc_call(arg);
        if (!alloc) has_restrict = false;
        auto p = allocs.insert(alloc);
        has_restrict &= p.second;
    }

    auto it_config = app->arg(KernelLaunchArgs::Config)->isa<Tuple>();
    if (it_config &&
        it_config->op(0)->isa<PrimLit>() &&
        it_config->op(1)->isa<PrimLit>() &&
        it_config->op(2)->isa<PrimLit>()) {
        return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{
                it_config->op(0)->as<PrimLit>()->qu32_value().data(),
                it_config->op(1)->as<PrimLit>()->qu32_value().data(),
                it_config->op(2)->as<PrimLit>()->qu32_value().data()
        }, has_restrict);
    }
    return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{-1, -1, -1}, has_restrict);
}

Backend::Backend(thorin::DeviceBackends& backends, World& src) : backends_(backends), device_code_(src), importer_(std::make_unique<Importer>(src, device_code_.world())) {}

struct CudaBackend : public Backend {
    explicit CudaBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::CUDA, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(device_code_, kernel_configs_, c::Lang::CUDA, backends_.debug(), empty);
    }
};

struct OpenCLBackend : public Backend {
    explicit OpenCLBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::OpenCL, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(device_code_, kernel_configs_, c::Lang::OpenCL, backends_.debug(), empty);
    }
};

#if THORIN_ENABLE_SPIRV
struct OpenCLSPIRVBackend : public Backend {
    explicit OpenCLSPIRVBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::OpenCL_SPIRV, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(device_code_, target, backends_.debug(), &kernel_configs_);
    }
};

struct LevelZeroSPIRVBackend : public Backend {
    explicit LevelZeroSPIRVBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::LevelZero_SPIRV, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(device_code_, target, backends_.debug(), &kernel_configs_);
    }
};
#endif

#if THORIN_ENABLE_LLVM
struct AMDHSABackend : public Backend {
    explicit AMDHSABackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::AMDGPUHSA, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUHSACodeGen>(device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }
};

struct AMDPALBackend : public Backend {
    explicit AMDPALBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::AMDGPUPAL, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUPALCodeGen>(device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }
};

struct NVVMBackend : public Backend {
    explicit NVVMBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::NVVM, *this, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::NVVMCodeGen>(device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }
};
#endif

#if THORIN_ENABLE_SHADY
struct ShadyBackend : public Backend {
    explicit ShadyBackend(DeviceBackends2& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::ShadyCompute, get_gpu_kernel_config);
    }

    std::unique_ptr<CodeGen> create_cg(const Cont2Config& config) override {
        return std::make_unique<shady_be::CodeGen>(device_code_, config, backends_.debug());
    }
};
#endif

struct HLSBackend : public Backend {
    explicit HLSBackend(DeviceBackends& b, World& src, std::string& hls_flags) : Backend(b, src), hls_flags_(hls_flags) {
        b.register_intrinsic(Intrinsic::HLS, *this, [&](const App* app, Continuation* kernel) {
            HLSKernelConfig::Param2Size param_sizes;
            for (size_t i = hls_free_vars_offset, e = app->num_args(); i != e; ++i) {
                auto arg = app->arg(i);
                auto ptr_type = arg->type()->isa<PtrType>();
                if (!ptr_type) continue;
                auto size = get_alloc_size(arg);
                if (size == 0)
                    b.world().edef(arg, "array size is not known at compile time");
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
                    b.world().edef(arg, "only pointers to arrays of primitive types are supported");
                auto num_elems = size / (multiplier * num_bits(prim_type->primtype_tag()) / 8);
                // imported has type: fn (mem, fn (mem), ...)
                param_sizes.emplace(kernel->param(i - hls_free_vars_offset + 2), num_elems);
            }
            return std::make_unique<HLSKernelConfig>(param_sizes);
        });
    }

    std::unique_ptr<CodeGen> create_cg() override {
        Top2Kernel top2kernel;
        DeviceParams hls_host_params;

        hls_host_params = hls_channels(device_code_, *importer_, top2kernel, backends_.world());
        hls_annotate_top(device_code_.world(), top2kernel, kernel_configs_);
        hls_kernel_launch(device_code_.world(), hls_host_params);

        return std::make_unique<c::CodeGen>(device_code_, kernel_configs_, c::Lang::HLS, backends_.debug(), hls_flags_);
    }

    std::string& hls_flags_;
};

DeviceBackends::DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags) : world_(world), opt_(opt), debug_(debug) {
    register_backend(std::make_unique<CudaBackend>(*this, world_));
    register_backend(std::make_unique<OpenCLBackend>(*this, world_));
#if THORIN_ENABLE_LLVM
    register_backend(std::make_unique<AMDHSABackend>(*this, world_));
    register_backend(std::make_unique<AMDPALBackend>(*this, world_));
    register_backend(std::make_unique<NVVMBackend>(*this, world_));
#endif
#if THORIN_ENABLE_SHADY
    register_backend(std::make_unique<ShadyBackend>(*this, world))
#endif
#if THORIN_ENABLE_SPIRV
    register_backend(std::make_unique<OpenCLSPIRVBackend>(*this, world));
    register_backend(std::make_unique<LevelZeroSPIRVBackend>(*this, world));
#endif
    register_backend(std::make_unique<HLSBackend>(*this, world_, hls_flags));

    lower_offload_intrinsics(world, *this);

    for (auto& backend : backends_) {
        if (backend->thorin().world().empty())
            continue;

        backend->prepare_kernel_configs();
        cgs.emplace_back(backend->create_cg());
    }
}

void DeviceBackends::register_backend(std::unique_ptr<Backend> backend) {
    backends_.push_back(std::move(backend));
}

World& DeviceBackends::world() { return world_; }
bool DeviceBackends::debug() { return debug_; }
int DeviceBackends::opt() { return opt_; }

void DeviceBackends::register_intrinsic(thorin::Intrinsic intrinsic, Backend& backend, GetKernelConfigFn f) {
    intrinsics_[intrinsic] = std::make_pair(&backend, f);
}

void DeviceBackends::register_kernel_for_offloading(const App* launch, Continuation* kernel) {
    Continuation* intrinsic_cont = launch->callee()->as_nom<Continuation>();
    auto handler = intrinsics_.find(intrinsic_cont->intrinsic());
    assert(handler != intrinsics_.end());
    auto [backend, get_config] = handler->second;

    // Import the continuation in the destination world
    Continuation* imported = backend->importer_->import(kernel)->as_nom<Continuation>();
    assert(imported);
    imported->world().make_external(imported);
    imported->attributes().cc = CC::C;

    // Obtain the kernel config now
    auto config = get_config(launch, kernel);
    backend->kernel_configs_[kernel] = std::move(config);
}

CodeGen::CodeGen(Thorin& thorin, bool debug)
    : thorin_(thorin)
    , debug_(debug)
{}

}
