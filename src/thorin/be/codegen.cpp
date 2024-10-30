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

namespace thorin {

void Backend::prepare_kernel_configs() {
    device_code_.opt();

    auto conts = device_code_.world().copy_continuations();
    for (auto continuation : kernels_) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto original_cont : conts) {
            if (!original_cont) continue;
            if (!original_cont->has_body()) continue;
            if (original_cont->name() == continuation->name())
                imported = original_cont;
        }
        if (!imported) continue;

        visit_uses(continuation, [&] (Continuation* use) {
            assert(use->has_body());

            auto handler = backends_.intrinsics_.find(use->body()->callee()->as<Continuation>()->intrinsic());
            assert(handler != backends_.intrinsics_.end());
            auto [backend2, get_config] = handler->second;
            assert(backend2 == this);

            auto config = get_config(use->body(), imported);
            if (config) {
                auto p = kernel_configs_.emplace(imported, std::move(config));
                assert_unused(p.second && "single kernel config entry expected");
            }
            return false;
        }, true);

        continuation->world().make_external(continuation);
        continuation->destroy("codegen");
    }
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

static std::unique_ptr<GPUKernelConfig> get_gpu_kernel_config(const App* app, Continuation* imported) {
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
        b.register_intrinsic(Intrinsic::HLS, *this, [&](const App* app, Continuation* imported) {
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
                param_sizes.emplace(imported->param(i - hls_free_vars_offset + 2), num_elems);
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

DeviceBackends::DeviceBackends(thorin::World& world, int opt, bool debug, std::string& hls_flags) : world_(world), opt_(opt), debug_(debug) {
    register_backend(std::make_unique<CudaBackend>(*this, world));
    register_backend(std::make_unique<OpenCLBackend>(*this, world));
#if THORIN_ENABLE_LLVM
    register_backend(std::make_unique<AMDHSABackend>(*this, world));
    register_backend(std::make_unique<AMDPALBackend>(*this, world));
    register_backend(std::make_unique<NVVMBackend>(*this, world));
#endif
#if THORIN_ENABLE_SHADY
    register_backend(std::make_unique<ShadyBackend>(*this, world))
#endif
#if THORIN_ENABLE_SPIRV
    register_backend(std::make_unique<OpenCLSPIRVBackend>(*this, world));
    register_backend(std::make_unique<LevelZeroSPIRVBackend>(*this, world));
#endif
    register_backend(std::make_unique<HLSBackend>(*this, world, hls_flags));

    search_for_device_code();
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

void DeviceBackends::search_for_device_code() {
    // determine different parts of the world which need to be compiled differently
    ScopesForest(world_).for_each([&] (const Scope& scope) {
        auto continuation = scope.entry();
        Continuation* imported = nullptr;

        Intrinsic intrinsic = Intrinsic::None;
        visit_capturing_intrinsics(continuation, [&] (Continuation* continuation) {
            if (continuation->is_accelerator()) {
                intrinsic = continuation->intrinsic();
                return true;
            }
            return false;
        });

        if (intrinsic == Intrinsic::None)
            return;

        auto handler = intrinsics_.find(intrinsic);
        assert(handler != intrinsics_.end());
        auto [backend, get_config] = handler->second;

        imported = backend->importer_->import(continuation)->as_nom<Continuation>();
        if (imported == nullptr)
            return;

        // Necessary so that the names match in the original and imported worlds
        imported->set_name(continuation->unique_name());
        continuation->set_name(continuation->unique_name());
        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->set_name(continuation->param(i)->name());
        imported->world().make_external(imported);
        imported->attributes().cc = CC::C;

        backend->kernels_.emplace_back(continuation);
    });

    for (auto& backend : backends_) {
        if (backend->thorin().world().empty())
            continue;

        backend->prepare_kernel_configs();
        cgs.emplace_back(backend->create_cg());
    }
}

CodeGen::CodeGen(Thorin& thorin, bool debug)
    : thorin_(thorin)
    , debug_(debug)
{}

}
