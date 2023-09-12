#include "thorin/be/codegen.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/transform/hls_kernel_launch.h"

#if THORIN_ENABLE_LLVM
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu.h"
#endif
#if THORIN_ENABLE_SHADY
#include "thorin/be/shady/shady.h"
#undef empty
#undef nodes
#endif
#include "thorin/be/c/c.h"

namespace thorin {

static void get_kernel_configs(
    Thorin& thorin,
    const std::vector<Continuation*>& kernels,
    Cont2Config& kernel_configs,
    std::function<std::unique_ptr<KernelConfig> (Continuation*, Continuation*)> use_callback)
{
    thorin.opt();

    auto externals = thorin.world().externals();
    for (auto continuation : kernels) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto [_, def] : externals) {
            auto exported = def->isa<Continuation>();
            if (!exported) continue;
            if (!exported->has_body()) continue;
            if (exported->name() == continuation->name())
                imported = exported;
        }
        if (!imported) continue;

        visit_uses(continuation, [&] (Continuation* use) {
            assert(use->has_body());
            auto config = use_callback(use, imported);
            if (config) {
                auto p = kernel_configs.emplace(imported, std::move(config));
                assert_unused(p.second && "single kernel config entry expected");
            }
            return false;
        }, true);

        continuation->attributes().cc = CC::DeviceHostCode;
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

DeviceBackends::DeviceBackends(World& world, int opt, bool debug, std::string& flags)
    : cgs {}
{
    std::vector<Importer> importers;
    for (auto& name : backend_names) {
        accelerator_code.emplace_back(name);
        importers.emplace_back(world, accelerator_code.back().world());
    }

    // determine different parts of the world which need to be compiled differently
    ScopesForest forest(world);
    forest.for_each([&] (const Scope& scope) {
        auto continuation = scope.entry();
        Continuation* imported = nullptr;

        static const auto backend_intrinsics = std::array {
            std::pair { CUDA,   Intrinsic::CUDA         },
            std::pair { NVVM,   Intrinsic::NVVM         },
            std::pair { OpenCL, Intrinsic::OpenCL       },
            std::pair { AMDGPU, Intrinsic::AMDGPU       },
            std::pair { HLS,    Intrinsic::HLS          },
            std::pair { Shady,  Intrinsic::ShadyCompute }
        };
        for (auto [backend, intrinsic] : backend_intrinsics) {
            if (is_passed_to_intrinsic(continuation, intrinsic)) {
                imported = importers[backend].import(continuation)->as_nom<Continuation>();
                break;
            }
        }

        if (imported == nullptr)
            return;

        // Necessary so that the names match in the original and imported worlds
        imported->set_name(continuation->unique_name());
        continuation->set_name(continuation->unique_name());
        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->set_name(continuation->param(i)->name());
        imported->world().make_external(imported);

        kernels.emplace_back(continuation);
    });

    for (auto backend : std::array { CUDA, NVVM, OpenCL, AMDGPU, Shady }) {
        if (!accelerator_code[backend].world().empty()) {
            get_kernel_configs(accelerator_code[backend], kernels, kernel_config, [&](Continuation *use, Continuation * /* imported */) {
                auto app = use->body();
                // determine whether or not this kernel uses restrict pointers
                bool has_restrict = true;
                DefSet allocs;
                for (size_t i = LaunchArgs::Num, e = app->num_args(); has_restrict && i != e; ++i) {
                    auto arg = app->arg(i);
                    if (!arg->type()->isa<PtrType>()) continue;
                    auto alloc = get_alloc_call(arg);
                    if (!alloc) has_restrict = false;
                    auto p = allocs.insert(alloc);
                    has_restrict &= p.second;
                }

                auto it_config = app->arg(LaunchArgs::Config)->as<Tuple>();
                if (it_config->op(0)->isa<PrimLit>() &&
                    it_config->op(1)->isa<PrimLit>() &&
                    it_config->op(2)->isa<PrimLit>()) {
                    return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{
                        it_config->op(0)->as<PrimLit>()->qu32_value().data(),
                        it_config->op(1)->as<PrimLit>()->qu32_value().data(),
                        it_config->op(2)->as<PrimLit>()->qu32_value().data()
                    }, has_restrict);
                }
                return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{-1, -1, -1}, has_restrict);
            });
        }
    }

    // get the HLS kernel configurations
    Top2Kernel top2kernel;
    DeviceParams hls_host_params;
    if (!accelerator_code[HLS].world().empty()) {
        hls_host_params = hls_channels(accelerator_code[HLS], importers[HLS], top2kernel, world);

        get_kernel_configs(accelerator_code[HLS], kernels, kernel_config, [&] (Continuation* use, Continuation* imported) {
            auto app = use->body();
            HLSKernelConfig::Param2Size param_sizes;
            for (size_t i = hls_free_vars_offset, e = app->num_args(); i != e; ++i) {
                auto arg = app->arg(i);
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
                param_sizes.emplace(imported->param(i - hls_free_vars_offset + 2), num_elems);
            }
            return std::make_unique<HLSKernelConfig>(param_sizes);
        });
        hls_annotate_top(importers[HLS].world(), top2kernel, kernel_config);
    }
    hls_kernel_launch(world, hls_host_params);

#if THORIN_ENABLE_LLVM
    if (!accelerator_code[NVVM  ].world().empty()) cgs[NVVM  ] = std::make_unique<llvm::NVVMCodeGen  >(accelerator_code[NVVM  ], kernel_config,      debug);
    if (!accelerator_code[AMDGPU].world().empty()) cgs[AMDGPU] = std::make_unique<llvm::AMDGPUCodeGen>(accelerator_code[AMDGPU], kernel_config, opt, debug);
#else
    (void)opt;
#endif
#if THORIN_ENABLE_SHADY
    if (!accelerator_code[Shady].world().empty()) cgs[Shady] = std::make_unique<shady_be::CodeGen>(accelerator_code[Shady], kernel_config, debug);
#endif
    for (auto [backend, lang] : std::array { std::pair { CUDA, c::Lang::CUDA }, std::pair { OpenCL, c::Lang::OpenCL }, std::pair { HLS, c::Lang::HLS } })
        if (!accelerator_code[backend].world().empty()) cgs[backend] = std::make_unique<c::CodeGen>(accelerator_code[backend], kernel_config, lang, debug, flags);
}

CodeGen::CodeGen(Thorin& thorin, bool debug)
    : thorin_(thorin)
    , debug_(debug)
{}

}
