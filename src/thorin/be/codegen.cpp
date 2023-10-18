#include "thorin/world.h"
#include "thorin/be/codegen.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/cgra_dataflow.h"

#if THORIN_ENABLE_LLVM
#include "thorin/be/llvm/cpu.h"
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu.h"
#endif
#include "thorin/be/c/c.h"
#include "thorin/be/config_script/config_script.h"

namespace thorin {

static void get_kernel_configs(
    Importer& importer,
    const std::vector<Continuation*>& kernels,
    Cont2Config& kernel_configs,
    std::function<std::unique_ptr<KernelConfig> (Continuation*, Continuation*)> use_callback,
    const std::function<void(const World::Externals&)>& cgra_callback = {})
{
    importer.world().opt();

    auto externals = importer.world().externals();
    if (cgra_callback)
        cgra_callback(externals);

    // accessd one time
    // add index to extract the port
    // make it a function like "kernel find by name"
   // for (auto [_, exported] : externals) {
   //     if (exported->name() == "hls_top") {
   //         std::cout << "I AM HLS_TOP" <<std::endl;
   //         for (auto param : exported->params()) {
   //             std::cout << "PARAM" <<std::endl;
   //             param->dump();
   //         }
   //     } else if (exported->name() == "cgra_graph"){

   //         std::cout << "I AM CGRA_GRAPH" <<std::endl;
   //         for (auto param : exported->params()) {
   //             std::cout << "PARAM" <<std::endl;
   //             param->dump();
   //         }

   //     }
   // }
    for (auto continuation : kernels) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto [_, exported] : externals) {
            if (!exported->has_body()) continue;
            if (exported->name() == continuation->unique_name())
                imported = exported;

        //    if (exported->name() == "hls_top") {
        //        std::cout << "I AM HLS_TOP" <<std::endl;
        //        for (auto param : exported->params()) {
        //            std::cout << "PARAM" <<std::endl;
        //            param->dump();
        //        }
        //    } else if (exported->name() == "cgra_graph"){

        //        std::cout << "I AM CGRA_GRAPH" <<std::endl;
        //        for (auto param : exported->params()) {
        //            std::cout << "PARAM" <<std::endl;
        //            param->dump();
        //        }

        //    }


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

//template<typename T>
//static bool has_restrict_pointer(const T device, Continuation* use) {
//    bool has_restrict = true;
//    auto app = use->body();
//    // determine whether or not this kernel uses restrict pointers
//    DefSet allocs;
//    for (size_t i = LaunchArgs<device>::Num, e = app->num_args(); has_restrict && i != e; ++i) {
//        auto arg = app->arg(i);
//        if (!arg->type()->isa<PtrType>()) continue;
//        auto alloc = get_alloc_call(arg);
//        if (!alloc) has_restrict = false;
//        auto p = allocs.insert(alloc);
//        has_restrict &= p.second;
//    }
//    return has_restrict;
//}
//
//
//
//
//

//static bool has_restrict_pointer(int launch_args_num, Continuation* use) {
static bool has_restrict_pointer(int launch_args_num, Continuation* use) {
// determines whether or not a kernel uses restrict pointers
    auto has_restrict = true;
    auto app = use->body();
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
//static const auto get_ports(const T param_status, const World::Externals& externals, HlsCgraPorts hls_cgra_ports = HlsCgraPorts()) {
static const auto get_ports(const T param_status, const World::Externals& externals, Ports& hls_cgra_ports) {
    for (auto [_, exported] : externals) {
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

//static const void get_ports_for(const std::string device_top, Array<size_t> param_indices, const World::Externals& externals) {
////static const void get_ports_for(const std::string device_top, PortStatus  param_indices, const World::Externals& externals) {
//    assert((device_top == "hls_top" || device_top == "cgra_graph") && "device top name is not valid!");
//    for (auto [_, exported] : externals) {
//        if (exported->name() == device_top) {
//            std::cout << "I am " << device_top <<std::endl;
//            for (auto param_index : param_indices) {
//                std::cout << "Ext port: ";
//                exported->param(param_index)->dump();
//                std::cout << exported->param(param_index)->unique_name() << std::endl;
//                //TODO: use index to check if a port is W or R. using global2mode or def2mde inside dataflow_HLS
//            }
//        }
//    }
//}
//
//
//
//
//static const void get_ports_for(const std::string device_top, PortStatus port_status, const World::Externals& externals) {
////static const void get_ports_for(const std::string device_top, PortStatus  param_indices, const World::Externals& externals) {
//    assert((device_top == "hls_top" || device_top == "cgra_graph") && "device top name is not valid!");
//    for (auto [_, exported] : externals) {
//        if (exported->name() == device_top) {
//            std::cout << "I am " << device_top <<std::endl;
//            for (auto [param_index, _] : port_status) {
//                std::cout << "Ext port: ";
//                exported->param(param_index)->dump();
//                std::cout << exported->param(param_index)->unique_name() << std::endl;
//                //TODO: use index to check if a port is W or R. using global2mode or def2mde inside dataflow_HLS
//            }
//        }
//    }
//}


////template<typename T>
//static bool has_restrict_pointer(Device_code device, Continuation* use) {
//    bool has_restrict = true;
//    auto app = use->body();
//    // determine whether or not this kernel uses restrict pointers
//    DefSet allocs;
//    //for (size_t i = LaunchArgs<device>::Num, e = app->num_args(); has_restrict && i != e; ++i) {
//    for (size_t i = launch_args(device)::Num, e = app->num_args(); has_restrict && i != e; ++i) {
//        auto arg = app->arg(i);
//        if (!arg->type()->isa<PtrType>()) continue;
//        auto alloc = get_alloc_call(arg);
//        if (!alloc) has_restrict = false;
//        auto p = allocs.insert(alloc);
//        has_restrict &= p.second;
//    }
//    return has_restrict;
//}

DeviceBackends::DeviceBackends(World& world, int opt, bool debug, std::string& flags)
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
            std::pair { HLS,    Intrinsic::HLS    },
            std::pair { CGRA,   Intrinsic::CGRA   }
        };
        for (auto [backend, intrinsic] : backend_intrinsics) {
            if (is_passed_to_intrinsic(continuation, intrinsic)) {
                imported = importers_[backend].import(continuation)->as_nom<Continuation>();
                break;
            }
        }

        if (imported == nullptr)
            return;

        // Necessary so that the names match in the original and imported worlds
        imported->set_name(continuation->unique_name());
        for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
            imported->param(i)->set_name(continuation->param(i)->name());
        imported->world().make_external(imported);

        kernels.emplace_back(continuation);
    });

    //for (auto backend : std::array { CUDA, NVVM, OpenCL, AMDGPU, CGRA }) {
    for (auto backend : std::array { CUDA, NVVM, OpenCL, AMDGPU}) {
        if (!importers_[backend].world().empty()) {
            //size_t launch_args_num;
           // switch (backend) {
             //   case CUDA: case NVVM: case OpenCL: case AMDGPU:
                    //launch_args_num = LaunchArgs<GPU>::Num; break;
              //  case CGRA: {
                    //cgra_dataflow(importers_[CGRA]);
              //      launch_args_num = LaunchArgs<AIE_CGRA>::Num; break;
        //        }
          //      default:
           //         THORIN_UNREACHABLE;
           // }

            get_kernel_configs(importers_[backend], kernels, kernel_config, [&](Continuation *use, Continuation * /* imported */) {
               // bool has_restrict = true;
                auto has_restrict = has_restrict_pointer(LaunchArgs<GPU>::Num, use);
                //auto app = use->body();
                // determine whether or not this kernel uses restrict pointers
            //    DefSet allocs;
            //    for (size_t i = launch_args_num, e = app->num_args(); has_restrict && i != e; ++i) {
            //        auto arg = app->arg(i);
            //        if (!arg->type()->isa<PtrType>()) continue;
            //        auto alloc = get_alloc_call(arg);
            //        if (!alloc) has_restrict = false;
            //        auto p = allocs.insert(alloc);
            //        has_restrict &= p.second;
            //    }
           // if (backend != CGRA) {
                auto it_config = use->body()->arg(LaunchArgs<GPU>::Config)->as<Tuple>();
                if (it_config->op(0)->isa<PrimLit>() &&
                    it_config->op(1)->isa<PrimLit>() &&
                    it_config->op(2)->isa<PrimLit>()) {
                    return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{
                        it_config->op(0)->as<PrimLit>()->qu32_value().data(),
                        it_config->op(1)->as<PrimLit>()->qu32_value().data(),
                        it_config->op(2)->as<PrimLit>()->qu32_value().data()
                    }, has_restrict);
                }
           // }
                return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{-1, -1, -1}, has_restrict);
            });
        }
    }

  //  if (!importers_[CGRA].world().empty()) {
  //      cgra_dataflow(importers_[CGRA]);
  //  }
    // get the HLS kernel configurations
    Top2Kernel top2kernel;
    DeviceDefs device_defs;
    Ports hls_cgra_ports; // channel-params between HLS and CGRA
    if (!importers_[HLS].world().empty()) {
        device_defs = hls_dataflow(importers_[HLS], top2kernel, world, importers_[CGRA]);

        get_kernel_configs(importers_[HLS], kernels, kernel_config, [&] (Continuation* use, Continuation* imported) {
            auto app = use->body();

           // auto externals = importers_[HLS].world().externals();
           // std::cout << "LOCAL CODE EXTERNALS" << std::endl;
           // for (auto [_, exported] : externals) {
           //     exported->dump();
           //     }


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
            return std::make_unique<HLSKernelConfig>(param_sizes); // this config is added into cont2config (kernel_config) map with its continuation
        }, [&] (const World::Externals& externals) {

            //auto externals = importers_[HLS].world().externals();
       //     for (auto [_, exported] : externals) {
       //         if (exported->name() == "hls_top") {
       //             std::cout << "I AM HLS_TOP" <<std::endl;
       //         //    for (auto param : exported->params()) {
       //         //        std::cout << "PARAM" <<std::endl;
       //         //        param->dump();
       //         //    }

       //             for (auto param_index : std::get<2>(device_defs)) {
       //                 std::cout << "CGRA port param: " << std::endl;
       //                 exported->param(param_index)->dump();
       //             }

       //         } else if (exported->name() == "cgra_graph"){

       //             std::cout << "I AM CGRA_GRAPH" <<std::endl;
       //             for (auto param : exported->params()) {
       //                 std::cout << "PARAM" <<std::endl;
       //                 param->dump();
       //             }

       //         }
       //     }

        //get_ports_for("hls_top", std::get<2>(device_defs), externals);
        get_ports(std::get<2>(device_defs), externals, hls_cgra_ports);
        }

            );
        hls_annotate_top(importers_[HLS].world(), top2kernel, kernel_config); // adding hls_top config to cont2config map
    }

    hls_kernel_launch(world, std::get<0>(device_defs));

    //TODO: need to write an analysis to check R/W mode on global memory allocaions
    if (!importers_[CGRA].world().empty()) {
        // at the moment only kernel channel modes are returned
        // ports are cgra_graph params that are connected to hls_top
       auto [port_indices, cont2param_modes] = cgra_dataflow(importers_[CGRA], world, std::get<1>(device_defs));

       get_kernel_configs(importers_[CGRA], kernels, kernel_config, [&] (Continuation* use, Continuation* imported) {
                CGRAKernelConfig::Param2Mode param2mode;

               // The order that channel modes are inserted in param_modes cosecuteviley is aligned with the order that channels appear in imported continuations
               // for example, the first mode in param_modes (index = 0) is equal to the first channel in the imported continuation (kernel)

               annotate_channel_modes(imported, cont2param_modes, param2mode);
               // for (auto const& [cont_name, param_modes] : cont2param_modes) {
               // if ( cont_name == imported->name()) {std::cout<< "BINGO!" << std::endl;
               // std::cout << "CONT2PARAM_MODES" << std::endl;
               // std::cout << cont_name << std::endl;
               // size_t index = 0;
               // for (auto const& param : imported->params()) {
               // if ((param->index() < 2) || is_mem(param) || param->order() != 0 || is_unit(param))
               // continue;
               // else if (auto type = param->type(); is_channel_type(type)) {
               // param2mode.emplace(param, param_modes[index++]);

               // }

               // break;
               // }
               // }
               // }


            auto app = use->body();
          //  for(const auto& [cont, param_modes] : cont2param_modes) {
          //  //TODO: check for continuation names then insert channel param modes
          //   //std::cout << "check names" << std::endl;
          //      //std::cout << cont->name() << "==" << imported->name() << " ?" << std::endl;
          //      //if (cont->name() == imported->name()) {std::cout << "BINGO" << std::endl;}
          //      for (auto const& param : imported->params()) {
          //          if ((param->index() < 2) || is_mem(param) || param->order() != 0 || is_unit(param))
          //              continue;
          //          else if (auto type = param->type(); is_channel_type(type)) {}

          //          param->dump();
          //      }
          //  }

            auto has_restrict = has_restrict_pointer(LaunchArgs<AIE_CGRA>::Num, use);
            // TODO: (-10,-10) auto location , default rtm_ratio to 1
                auto runtime_ratio = app->arg(LaunchArgs<AIE_CGRA>::RUNTIME_RATIO);
                auto tile_location = app->arg(LaunchArgs<AIE_CGRA>::LOCATION)->as<Tuple>();
                if (runtime_ratio->isa<PrimLit>() &&
                    tile_location->op(0)->isa<PrimLit>() &&
                    tile_location->op(1)->isa<PrimLit>()) {
                    auto runtime_ratio_val = runtime_ratio->as<PrimLit>()->qf32_value().data();
                    auto tile_location_val = std::make_pair(tile_location->op(0)->as<PrimLit>()->qu32_value().data(),
                           tile_location->op(1)->as<PrimLit>()->qu32_value().data());

                    return std::make_unique<CGRAKernelConfig>(runtime_ratio_val, tile_location_val, param2mode, has_restrict);
                }
                    return std::make_unique<CGRAKernelConfig>(-1, std::make_pair(-1, -1), param2mode, has_restrict);

            // TODO: insert corresponding params from imported  using index and add mode
            //    for (size_t i = cgra_free_vars_offset, e = app->num_args(); i != e; ++i) {
            //        auto arg = app->arg(i);
            //        auto ptr_type = arg->type()->isa<PtrType>();
            //        //TODO : check types and assign to param2mode
            //    }
        }, [&] (const World::Externals& externals) {
            // TODO: HERE cgra_graph params are correct!
            Continuation* cgra_graph_cont = nullptr;
            for (auto [_, exported] : externals) {
                if (auto temp = exported->isa_nom<Continuation>()) {
                    std::cout << "external codegen" << std::endl;
                    temp->dump();
                }
                if (exported->isa_nom<Continuation>()->is_cgra_graph()) {
                    cgra_graph_cont = exported;
                    for (auto param : exported->params()){
                        std::cout << "external" << std::endl;
                        param->dump();
                    }
                }
            }
      //  }

                //get_ports_for("cgra_graph", port_indices, externals);
                //get_ports(std::get<0>(port_indices), externals, hls_cgra_ports);
                get_ports(port_indices, externals, hls_cgra_ports);
        // just copied from down
   //     if (!hls_cgra_ports.empty()) {
        // the aim is passing cgra_graph cont to annotate_cgra_graph_modes to import the config for non-channel params
            cgra_graph_cont->dump();
            annotate_cgra_graph_modes(cgra_graph_cont, hls_cgra_ports, kernel_config); // adding cgra_graph config to cont2config map
            for (const auto& item : kernel_config) {
                auto cont = item.first;
                if (auto config = item.second->isa<CGRAKernelConfig>(); config) {std::cout << "FOUND CGRA CONFIG" << std::endl;
                    cont->dump();
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
      //  }
        //else
          //  world.WLOG("TODO: CGRA graph is not correclty generated due to direct memory access!");

        }
        );
        // just moved up but here is a better place as it is called only once
//        if (!hls_cgra_ports.empty()) {
//            annotate_cgra_graph_modes(hls_cgra_ports, kernel_config); // adding cgra_graph config to cont2config map
//            for (const auto& item : kernel_config) {
//                auto cont = item.first;
//                if (auto config = item.second->isa<CGRAKernelConfig>(); config) {std::cout << "FOUND CGRA CONFIG" << std::endl;
//                    cont->dump();
//                    for (auto param : cont->params()) {
//                        if (auto mode = config->param_mode(param); mode != ChannelMode::Undef) {
//                            std::cout << "param" << std::endl;
//                            param->dump();
//                            std::cout << "mode" << std::endl;
//                            if (mode == ChannelMode::Read) {std::cout << "Read"<< std::endl;
//                            } else {
//                                std::cout << "Write"<< std::endl;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//        else
//            world.WLOG("TODO: CGRA graph is not correclty generated due to direct memory access!");
    }


#if THORIN_ENABLE_LLVM
    if (!importers_[NVVM  ].world().empty()) cgs[NVVM  ] = std::make_unique<llvm::NVVMCodeGen  >(importers_[NVVM  ].world(), kernel_config,      debug);
    if (!importers_[AMDGPU].world().empty()) cgs[AMDGPU] = std::make_unique<llvm::AMDGPUCodeGen>(importers_[AMDGPU].world(), kernel_config, opt, debug);
#else
    (void)opt;
#endif

        //thorin::config_script::CodeGen cg(world,debug, hls_cgra_ports);
        //emit_to_file(cg);
   // std::cout << "--------> " <<hls_cgra_ports.size() << std::endl;
   // for (auto elem : hls_cgra_ports)
       // std::cout << "--------> " << elem.first.value().first << "-----" << elem.second.value() << std::endl;
        //elem.second.value();
    if (!importers_[CGRA].world().empty()){
        cgs[CGRA] = std::make_unique<config_script::CodeGen>(importers_[CGRA].world(), debug, hls_cgra_ports, flags);
        thorin::config_script::CodeGen cg(world, debug, hls_cgra_ports, flags);

    auto emit_to_file = [&] (thorin::CodeGen& cg) {
            auto name = world.name() + cg.file_ext();
            std::ofstream file(name);
            if (!file)
                world.ELOG("cannot open '{}' for writing", name);
            else
                cg.emit_stream(file);
        };

        emit_to_file(cg);
    }

    //if (!importers_[CGRA  ].world().empty()) cgs[CGRA  ] = std::make_unique<config_script::CodeGen>(importers_[CGRA  ].world(), debug);
    for (auto [backend, lang] : std::array { std::pair { CUDA, c::Lang::CUDA }, std::pair { OpenCL, c::Lang::OpenCL }, std::pair { HLS, c::Lang::HLS }, std::pair { CGRA, c::Lang::CGRA } })
        if (!importers_[backend].world().empty()) { cgs[backend] = std::make_unique<c::CodeGen>(importers_[backend].world(), kernel_config, lang, debug, flags);
            //importers_[HLS].world().dump();
        }

}



CodeGen::CodeGen(World& world, bool debug)
    : world_(world)
    , debug_(debug)
{}

}
