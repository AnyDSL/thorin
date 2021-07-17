#include "thorin/be/codegen.h"
#include "thorin/analyses/scope.h"

#if THORIN_ENABLE_LLVM
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu.h"
#endif
#include "thorin/be/c/c.h"

namespace thorin {

static void get_kernel_configs(
    Importer& importer,
    const std::vector<Continuation*>& kernels,
    Cont2Config& kernel_configs,
    std::function<std::unique_ptr<KernelConfig> (Continuation*, Continuation*)> use_callback)
{
    importer.world().opt();

    auto exported_continuations = importer.world().exported_continuations();
    for (auto continuation : kernels) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto exported : exported_continuations) {
            if (exported->name() == continuation->unique_name())
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
    if (ret->num_uses() != 1) return nullptr;

    auto use = *(ret->uses().begin());
    auto call = use.def()->isa<App>();
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

DeviceBackends::DeviceBackends(World& world, int opt, bool debug)
    : cgs {}
{
    for (size_t i = 0; i < cgs.size(); ++i)
        importers_.emplace_back(world);

    // determine different parts of the world which need to be compiled differently
    Scope::for_each(world, [&] (const Scope& scope) {
        auto continuation = scope.entry();
        Continuation* imported = nullptr;

        static const auto backend_intrinsics = std::array {
            std::pair { CUDA,   Intrinsic::CUDA   },
            std::pair { NVVM,   Intrinsic::NVVM   },
            std::pair { OpenCL, Intrinsic::OpenCL },
            std::pair { AMDGPU, Intrinsic::AMDGPU },
            std::pair { HLS,    Intrinsic::HLS    }
        };
        for (auto [backend, intrinsic] : backend_intrinsics) {
            if (is_passed_to_intrinsic(continuation, intrinsic)) {
                imported = importers_[backend].import(continuation)->as_continuation();
                break;
            }
        }

        if (imported == nullptr)
            return;

        // Necessary so that the names match in the original and imported worlds
        imported->set_name(continuation->unique_name());
        imported->make_external();
        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->set_name(continuation->param(i)->name());

        kernels.emplace_back(continuation);
    });

    for (auto backend : std::array { CUDA, NVVM, OpenCL, AMDGPU }) {
        if (!importers_[backend].world().empty()) {
            get_kernel_configs(importers_[backend], kernels, kernel_config, [&](Continuation *use, Continuation * /* imported */) {
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
    if (!importers_[HLS].world().empty()) {
        get_kernel_configs(importers_[HLS], kernels, kernel_config, [&] (Continuation* use, Continuation* imported) {
            auto app = use->body();
            HLSKernelConfig::Param2Size param_sizes;
            for (size_t i = 3, e = app->num_args(); i != e; ++i) {
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
                param_sizes.emplace(imported->param(i - 3 + 2), num_elems);
            }
            return std::make_unique<HLSKernelConfig>(param_sizes);
        });
    }

#if THORIN_ENABLE_LLVM
    if (!importers_[NVVM  ].world().empty()) cgs[NVVM  ] = std::make_unique<llvm::NVVMCodeGen  >(importers_[NVVM  ].world(), kernel_config,      debug);
    if (!importers_[AMDGPU].world().empty()) cgs[AMDGPU] = std::make_unique<llvm::AMDGPUCodeGen>(importers_[AMDGPU].world(), kernel_config, opt, debug);
#else
    (void)opt;
#endif
    for (auto [backend, lang] : std::array { std::pair { CUDA, c::Lang::CUDA }, std::pair { OpenCL, c::Lang::OpenCL }, std::pair { HLS, c::Lang::HLS } })
        if (!importers_[backend].world().empty()) cgs[backend] = std::make_unique<c::CodeGen>(importers_[backend].world(), kernel_config, lang, debug);
}

CodeGen::CodeGen(World& world, bool debug)
    : world_(world)
    , debug_(debug)
{}

}
