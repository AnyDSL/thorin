#include "thorin/world.h"
#include "thorin/be/codegen.h"

#include "thorin/be/c/c.h"
#include "thorin/be/config_script/config_script.h"
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

#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/cgra_dataflow.h"

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
            if (!original_cont->is_exported()) continue;
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

static bool has_restrict_pointer(int launch_args_num, const App* app) {
    // determines whether or not a kernel uses restrict pointers
    auto has_restrict = true;
    DefSet allocs;
    for (size_t i = launch_args_num, e = app->num_args(); has_restrict && i != e; ++i) {
        auto arg = app->arg(i);
        if (!arg->type()->isa<PtrType>()) continue;
        auto alloc = get_alloc_call(arg);
        if (!alloc) has_restrict = false;
        auto p = allocs.insert(alloc);
        has_restrict &= p.second;
    }
    return has_restrict;
}

// the order that indices appear in the hls and cgra arrays (port_status) are consistent
// meaning that they should get connected to each other
// It is true beacause of the design of the data structure
// for example the hls_top param with index 2 at position 1 and cgra_graph param with index 3 at position 1 of the array are semantically related.
template<typename T>
static const auto get_ports(const T param_status, const World::Externals& externals, Ports& hls_cgra_ports) {
    for (auto [_, exported_def] : externals) {
        auto exported = exported_def->isa<Continuation>();
        if (!exported) continue;
        //if (exported->name() == "hls_top" || exported->name() == "cgra_graph" ) {
        if (exported->is_hls_top() || exported->is_cgra_graph() ) {
            if constexpr (std::is_same_v<T, Array<size_t>>) {
                //CGRA
                if (hls_cgra_ports.empty()) {
                    for (auto param_index : param_status) {
                        //hls_cgra_ports.emplace_back(std::nullopt, exported->param(param_index)->unique_name());
                        hls_cgra_ports.emplace_back(std::nullopt, exported->param(param_index));
                    }
                } else {
                    for (size_t i = 0; i < hls_cgra_ports.size(); ++i) {
                        auto& [_, cgra_port_name] = hls_cgra_ports[i];
                        auto param_index = param_status[i];
                        //cgra_port_name = exported->param(param_index)->unique_name();
                        cgra_port_name = exported->param(param_index);
                    }
                }
                return;
            } else {
                //HLS
                if (hls_cgra_ports.empty()) {
                    for (auto [index, mode] : param_status) {
                        //hls_cgra_ports.emplace_back(std::make_pair(exported->param(index)->unique_name(), mode), std::nullopt);
                        hls_cgra_ports.emplace_back(std::make_pair(exported->param(index), mode), std::nullopt);
                    }
                } else {
                    for (size_t i = 0; i < hls_cgra_ports.size(); ++i) {
                        auto& [status, _]  = hls_cgra_ports[i];
                        auto [index, mode] = param_status[i];
                        //status = std::make_pair(exported->param(index)->unique_name(), mode);
                        status = std::make_pair(exported->param(index), mode); // status is a ref in hls_cgra_port (rewrites the value in hls_cgra_port)
                    }
                }
            }
            return;
        }
    }
    assert(false && "No top module found!");
}

static std::unique_ptr<GPUKernelConfig> get_gpu_kernel_config(const App* app, Continuation* imported) {
    bool has_restrict = has_restrict_pointer(KernelLaunchArgs<GPU>::Num, app);

    auto it_config = app->arg(KernelLaunchArgs<GPU>::Config)->isa<Tuple>();
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

//TODO: move this somewhere sane.
static Ports hls_cgra_ports; // chanel-params between HLS and CGRA
static DeviceDefs hls_device_defs;
static Top2Kernel hls_top2kernel;

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
        if (!device_code_.world().empty()) {
            get_ports(std::get<2>(hls_device_defs), device_code_.world().externals(), hls_cgra_ports);

            hls_annotate_top(device_code_.world(), hls_top2kernel, kernel_configs_);
            hls_kernel_launch(device_code_.world(), std::get<0>(hls_device_defs), kernel_configs_);
        }

        return std::make_unique<c::CodeGen>(device_code_, kernel_configs_, c::Lang::HLS, backends_.debug(), hls_flags_);
    }

    std::string& hls_flags_;
};

static ContName2ParamModes cgra_cont2param_modes;
static PortIndices cgra_port_indices;

struct CGRABackend : public Backend {
    explicit CGRABackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::CGRA, *this, [&](const App* app, Continuation* imported) {
            CGRAKernelConfig::Param2Mode param2mode;
            // The order that channel modes are inserted in param_modes cosecuteviley is aligned with the order that channels appear in imported continuations
            // for example, the first mode in param_modes (index = 0) is equal to the first channel in the imported continuation (kernel)

            annotate_channel_modes(imported, cgra_cont2param_modes, param2mode);
            for (auto use : app->uses()) {
                if (use->isa<Continuation>())
                    annotate_interface(imported, use->as<Continuation>());
            }

            auto has_restrict = has_restrict_pointer(KernelLaunchArgs<AIE_CGRA>::Num, app);
            // TODO: (-10,-10) auto location , default rtm_ratio to 1
            auto runtime_ratio = app->arg(KernelLaunchArgs<AIE_CGRA>::Runtime_ratio);
            auto tile_location = app->arg(KernelLaunchArgs<AIE_CGRA>::Location)->as<Tuple>();
            auto vector_size   = app->arg(KernelLaunchArgs<AIE_CGRA>::Vector_size);
            if (runtime_ratio->isa<PrimLit>() &&
                    tile_location->op(0)->isa<PrimLit>() &&
                    tile_location->op(1)->isa<PrimLit>() &&
                    vector_size->isa<PrimLit>()) {
                auto runtime_ratio_val = runtime_ratio->as<PrimLit>()->qf32_value().data();
                auto tile_location_val = std::make_pair(tile_location->op(0)->as<PrimLit>()->qu32_value().data(),
                        tile_location->op(1)->as<PrimLit>()->qu32_value().data());
                auto vector_size_val = vector_size->as<PrimLit>()->qu32_value().data();

                return std::make_unique<CGRAKernelConfig>(runtime_ratio_val, tile_location_val, vector_size_val, param2mode, has_restrict);
            }
            return std::make_unique<CGRAKernelConfig>(-1, std::make_pair(-1, -1), -1, param2mode, has_restrict);
        });
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;

        if (!device_code_.world().empty()) {
            // TODO: HERE cgra_graph params are correct!
            Continuation* cgra_graph_cont = nullptr;
            for (auto [_, exported] : device_code_.world().externals()) {
                if (auto cont = exported->isa_nom<Continuation>(); cont && cont->is_cgra_graph()) {
                    cgra_graph_cont = cont;
                    for (auto param : cont->params()){
                        std::cout << "external" << std::endl;
                        //param->dump();
                    }
                }
            }

            get_ports(cgra_port_indices, device_code_.world().externals(), hls_cgra_ports);
            annotate_cgra_graph_modes(cgra_graph_cont, hls_cgra_ports, kernel_configs_); // adding cgra_graph config to cont2config map
                                                                                         //
            for (const auto& item : kernel_configs_) {
                auto cont = item.first;
                if (auto config = item.second->isa<CGRAKernelConfig>(); config) {//std::cout << "FOUND CGRA CONFIG" << std::endl;
                                                                                 //cont->dump();
                    for (auto param : cont->params()) {
                        if (auto mode = config->param_mode(param); mode != ChannelMode::Undef) {
                            std::cout << "param" << std::endl;
                            param->dump();
                            std::cout << "mode" << std::endl;
                            if (mode == ChannelMode::Read) {std::cout << "Read"<< std::endl;
                            } else {
                                std::cout << "Write"<< std::endl;
                            }
                        }
                    }
                }
            }

            //TODO: This should not be emitted here. Eiter find a way to move this code generator outside, or integrate this into the CGRA code generator.
            //Maybe a wrapper around these code generators would be in order? We currently expect a single code generator to emit a single file,
            //but this is not the case with CGRA.
            auto cfg_generator = std::make_unique<config_script::CodeGen>(device_code_, backends_.debug(), hls_cgra_ports, empty);

            auto name = device_code_.world().name() + cfg_generator->file_ext();
            std::ofstream file(name);
            if (!file)
                device_code_.world().ELOG("cannot open '{}' for writing", name);
            else
                cfg_generator->emit_stream(file);
        }

        return std::make_unique<c::CodeGen>(device_code_, kernel_configs_, c::Lang::CGRA, backends_.debug(), empty);
    }
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
    register_backend(std::make_unique<CGRABackend>(*this, world));

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
            if (continuation->is_offload_intrinsic()) {
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

    auto [hls_backend, _] = intrinsics_.find(Intrinsic::HLS)->second;
    auto [cgra_backend, _] = intrinsics_.find(Intrinsic::CGRA)->second;

    hls_device_defs = hls_dataflow(*hls_backend->importer_, hls_top2kernel, world_, *cgra_backend->importer_);
    auto dataflow_result = cgra_dataflow(*cgra_backend->importer_, world_, std::get<1>(hls_device_defs));
    cgra_port_indices = std::get<0>(dataflow_result);
    cgra_cont2param_modes = std::get<1>(dataflow_result);

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
