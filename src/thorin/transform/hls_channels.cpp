#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

//using Def2Mode = DefMap<ChannelMode>;
using Dependencies = std::vector<std::pair<size_t, size_t>>; // (From, To)
using Def2Block = DefMap<std::pair<Continuation*, Intrinsic>>; // [global_def , (basicblock, HLS/CGRA intrinsic)]
using Def2DependentBlocks = DefMap<std::pair<Continuation*, Continuation*>>; // [global_def, (HLS_basicblock, CGRA_basicblock)]
std::vector<const Def*> hls_global;
std::vector<const Def*> cgra_global;


void hls_cgra_global_analysis(World& world, std::vector<Def2Block>& old_global_maps) {
    Scope::for_each(world, [&] (Scope& scope) {
            std::cout<< "*** SCOPE entry*** " <<std::endl;
            scope.entry()->dump();
            auto kernel = scope.entry();
            // Decide about scope intrinsic here by seach or use a stack on main scope!
             //for (auto n : scope.f_cfg().post_order()) { n->continuation()->dump();}
            //std::cout<< "*** blocks *** " <<std::endl;
            Def2Block global2block; // global, using basic block, HLS/CGRA
            for (auto& block : schedule(scope)) {
                if (!block->has_body())
                    continue;
                // block->dump();
                assert(block->has_body());
                auto body = block->body();
                auto callee = body->callee()->isa_nom<Continuation>();
                if (callee && callee->is_channel()) {
                    if (body->arg(1)->order() == 0 && !(is_mem(body->arg(1)) || is_unit(body->arg(1)))) {
                        std::cout<< "*** target callee *** " <<std::endl;
                        // global is here
                       // TODO: more checks on global def
                        auto def = body->arg(1);
                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                            callee->dump();
                        //   std::cout << "~~~~~~~~Global uses~~~~~~" << std::endl;
                        //   for (auto use : body->arg(1)->uses())
                        //       use->dump();
                            for (auto preds_scope : scope.entry()->preds()) {
                            //auto pred_scope_callee = preds_scope->body()->callee()->isa_nom<Continuation>();
                                if (auto pred_scope_callee = preds_scope->body()->callee()->isa_nom<Continuation>(); pred_scope_callee
                                        && pred_scope_callee->is_intrinsic()) {
                                    if (pred_scope_callee->intrinsic() == Intrinsic::HLS ||
                                            pred_scope_callee->intrinsic() == Intrinsic::CGRA) {
                                        std::cout << "~~~~~~~~Pred callee~~~~~~" << std::endl;
                                        pred_scope_callee->dump();
                                        global2block.emplace(def, std::make_pair(callee, pred_scope_callee->intrinsic()));
                                    }

                                }
                            }
                        }
                    }
                }


           //TODO:  scope is static in all corresponding basic blocks,
           //It seems the only way is searching over all defs and look for scope.entry in 2nd arg of callee
           // in fact the scope.entry should be equal to --> 2nd arg of HLS/CGRA callee->as Global->init
           // then the HLS/CGRA intrinsic can mark a flag for all basic blocks in that scope
            //body->dump();
            //auto scope_callee = scope.entry()->body()->callee();
      //      std::cout<< "*** SCOPE *** " <<std::endl;
      //      scope.entry()->dump();
      //      //scope_callee->dump();
      //      std::cout<< "*** Defs *** " <<std::endl;
      //      for (auto def : scope.defs())
      //          def->dump();
                //TODO: look for read and write in defs of scopes then look at the scope then search inside all block to find that scope then look at the corresponding callee to see whether it is HLS or CGRA

            auto callee_ = body->callee()->isa_nom<Continuation>();
            if (callee_ && callee_->intrinsic() == Intrinsic::CGRA) {
                auto cont = body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
                auto callee_ = cont->body()->callee()->isa_nom<Continuation>(); 
                if (callee_ && callee_->is_channel()) {
                    if (cont->body()->arg(1)->order() == 0 && !(is_mem(cont->body()->arg(1)) || is_unit(cont->body()->arg(1)))) {
                        auto def = cont->body()->arg(1);
                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                            cgra_global.emplace_back(def);
                            std::cout << "***cgra size: "  <<  cgra_global.size()<<"******" << std::endl;
                        }
                    }
                }
            }
            if (callee_ && callee_->intrinsic() == Intrinsic::HLS) {
                auto cont = body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
                auto callee_ = cont->body()->callee()->isa_nom<Continuation>();
                if (callee_ && callee_->is_channel()) {
                    if (cont->body()->arg(1)->order() == 0 && !(is_mem(cont->body()->arg(1)) || is_unit(cont->body()->arg(1)))) {
                        auto def = cont->body()->arg(1);
                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                //if (cont->body()->callee()->is_channel())
                           // for (auto use : def->uses()) {
                           //     //auto test = use->isa<App>();
                           //     auto test = use->isa<App>();
                           //     if (test) {
                           //         auto conts = test->using_continuations();
                           //         for (auto cont : conts)
                           //             cont->dump();
                           //     }
                           // }
                                //cont->dump(); TODO: we need to find corresponding continuations in the HLS world
                                //importer.def_old2new_[
                                hls_global.emplace_back(def);
                        }
                    }
                }
        }
        }
    if (!global2block.empty()) {
        // size of this vector should be the same as the number of kernels in HLS world
        // TODO: think about if repeating globals in different bb is required!
        old_global_maps.emplace_back(global2block);
        std::cout << " kernel number = " << old_global_maps.size() << std::endl;
        std::cout << " old_map_size = " << global2block.size() << std::endl;
        std::cout << " ####inside map #####" << std::endl;
        for (auto [k, v] : global2block)
            k->dump();

    }


          //  std::cout<< "*** scope conts *** " <<std::endl;
          //  for (auto def : scope.defs())
          //      if (auto cont = def->isa<Continuation>())
          //          cont->dump();
    });
    }

bool CheckCommon(std::vector<const Def*> const& inVectorA, std::vector<const Def*> const& nVectorB) {
    return std::find_first_of (inVectorA.begin(), inVectorA.end(),
            nVectorB.begin(), nVectorB.end()) != inVectorA.end();
}



// HLS-CGRA dependency search algorithm
void hls_cgra_dependency_analysis(Def2DependentBlocks& global2dependent_blocks, const std::vector<Def2Block>& old_global_maps) {
    std::vector<const Def*> visited_globals;
    for (auto cur_kernel_it = old_global_maps.cbegin(); cur_kernel_it != old_global_maps.cend(); ++cur_kernel_it) {
        auto cur_map = *cur_kernel_it;
        for (const auto& [cur_global, cur_pair] : cur_map) {
            // Filtereing out already visited globals from the search space
            if (std::find(visited_globals.cbegin(), visited_globals.cend(), cur_global) != visited_globals.cend())
                continue;
            auto [cur_basic_block, cur_intrinsic] = cur_pair;
            for(auto next_kernel_it = cur_kernel_it + 1; next_kernel_it != old_global_maps.cend(); ++next_kernel_it){
                auto next_map = *next_kernel_it;
                if (auto same_global_it = next_map.find(cur_global); same_global_it != next_map.end()) {
                    //found
                    auto [next_basic_block, next_intrinsic] = same_global_it->second;
                    if (cur_intrinsic != next_intrinsic) {
                        // HLS-CGR
                        // We assume that the current basic block blongs to HLS
                        auto hls_basic_block  = cur_basic_block;
                        auto cgra_basic_block = next_basic_block;
                        if (cur_intrinsic == Intrinsic::CGRA)
                            std::swap(hls_basic_block, next_basic_block);
                        global2dependent_blocks.emplace(cur_global, std::make_pair(hls_basic_block, cgra_basic_block));
                        break;
                    } else {
                        //HLS-HLS or CGRA-CGRA dependencies are filtered out
                        continue;
                    }
                }
            }
            visited_globals.emplace_back(cur_global);
        }
    }
}



//static void extract_kernel_channels(const Schedule& schedule, Def2Mode& def2mode) {
void extract_kernel_channels(const Schedule& schedule, Def2Mode& def2mode) {
    for (const auto& continuation : schedule) {

        if (!continuation->has_body())
            continue;
        auto app = continuation->body();

        auto callee = app->callee()->isa_nom<Continuation>();
        if (callee && callee->is_channel()) {
            if (app->arg(1)->order() == 0 && !(is_mem(app->arg(1)) || is_unit(app->arg(1)))) {
                auto def = app->arg(1);
                if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                    if (callee->name().find("write_channel") != std::string::npos) {
                        assert((!def2mode.contains(def) || def2mode[def] == ChannelMode::Write) &&
                                "Duplicated channel or \"READ\" mode channel redefined as WRITE!");
                        def2mode.emplace(def, ChannelMode::Write);
                    } else if (callee->name().find("read_channel") != std::string::npos) {
                        assert((!def2mode.contains(def) || def2mode[def] == ChannelMode::Read)  &&
                                "Duplicated channel or \"WRITE\" mode channel redefined as READ!");
                        def2mode.emplace(def, ChannelMode::Read);
                    } else {
                        continuation->world().ELOG("Not a channel / unsupported channel placeholder");
                    }
                }
            }
        }
    }
}


//TODO: Extract channels used in CGRAs

bool is_channel_type(const Type* type) {
    if (auto ptr_type = type->isa<PtrType>()) {
        if (auto struct_type = ptr_type->pointee()->isa<StructType>()) {
            if (struct_type->name().str().find("channel") != std::string::npos)
                return true;
        }
    }
    return false;
}

bool is_single_kernel(Continuation* kernel) {
    for (auto param : kernel->params()) {
        if (is_channel_type(param->type()))
            return false;
    }
    return true;
}

void hls_annotate_top(World& world, const Top2Kernel& top2kernel, Cont2Config& cont2config) {
    auto find_kernel_by_name = [&] (const std::string& name) {
        for (auto def : world.defs()) {
            auto continuation = def->isa_nom<Continuation>();
            if (!continuation) continue;
            if (continuation->is_exported() && continuation->name() == name)
                return continuation;
        }
        return (Continuation*)nullptr;
    };
    // Extract and save param size info for hls_top then insert it into configuration map.
    auto hls_top = find_kernel_by_name("hls_top");
    assert(hls_top);
    HLSKernelConfig::Param2Size param_sizes;
    for (auto& tuple : top2kernel) {
        auto& name = std::get<1>(tuple);
        auto kernel = find_kernel_by_name(name);
        assert(kernel && "where did my kernel go");
        auto param  = kernel->param(std::get<2>(tuple));
        auto config = cont2config[kernel]->as<HLSKernelConfig>();
        param_sizes[hls_top->param(std::get<0>(tuple))] = config->param_size(param);
    }
    cont2config.emplace(hls_top, std::make_unique<HLSKernelConfig>(param_sizes));
}

// ----------- Kernel scheduling (dependency resolver) algorithm -------------

// Find out if a kernel has no dependency (no input from other kernels to this one)
static bool is_free_kernel(const Dependencies& dependencies, const std::vector<bool>& dependency_bool_vector, const size_t kernel) {
    bool free_kernel = true;
    for (size_t i = 0; i < dependencies.size() && free_kernel; ++i) {
        free_kernel = (!dependency_bool_vector[i])|| (dependencies[i].second != kernel);
    }
    return free_kernel;
}

// Get the kernels with no dependency (no input from other kernels to those)
static size_t get_free_kernels(const Dependencies& dependencies, const std::vector<bool>& dependency_bool_vector,
        const size_t dependent_kernels_size, std::stack<size_t>& free_kernels) {
    for (size_t kernel = 0; kernel < dependent_kernels_size; ++kernel) {
        if (is_free_kernel(dependencies, dependency_bool_vector, kernel)) {
            free_kernels.push(kernel);
        }
    }
    return free_kernels.size();
}

bool dependency_resolver(Dependencies& dependencies, const size_t dependent_kernels_size, std::vector<size_t>& resolved) {
    std::stack<size_t> free_kernels;
    std::vector<bool> dependency_bool_vector(dependencies.size());
    size_t remaining_dependencies = dependencies.size();


    // in the begining all dependencies are marked
    for (size_t i = 0; i < dependencies.size(); ++i )
        dependency_bool_vector[i] = true;
    // Get the kernels with no incoming dependencies
    auto num_of_free_kernels = get_free_kernels(dependencies, dependency_bool_vector, dependent_kernels_size, free_kernels);
    // Main loop
    while (num_of_free_kernels) {
        // get a free kernel
        auto free = free_kernels.top();
        // Add it to the resolved array
        resolved.emplace_back(free);
        // Remove from free_kernels stack
        free_kernels.pop();
        // Remove all dependencies with other kernels
        for (size_t i = 0; i < dependencies.size(); ++i) {
            if (dependency_bool_vector[i] && dependencies[i].first == free) {
                dependency_bool_vector[i] = false;
                remaining_dependencies--;

                // Check if other kernels are free now
                if (is_free_kernel(dependencies, dependency_bool_vector, dependencies[i].second)) {
                    // Add it to set of free kernels
                    free_kernels.push(dependencies[i].second);
                }
            }
        }
        num_of_free_kernels = free_kernels.size();
    }
    // if there is no more free kernels but there exist dependencies, a cycle is found
    return remaining_dependencies == 0;
}

bool has_cgra_callee(World& world) {
    auto found_cgra = false;
    Scope::for_each(world, [&] (Scope& scope) {
        for (auto& block : schedule(scope)) {
            if (!block->has_body())
                continue;
            assert(block->has_body());
            auto body = block->body();
            auto callee = body->callee()->isa_nom<Continuation>();
           // if (callee && callee->is_channel()) {
           //     std::cout<< "channel" << std::endl;
           //     std::cout << "name-->" << callee->name()<< std::endl;
           //     body->dump();
           // }
            if (callee && callee->intrinsic() == Intrinsic::CGRA) {
                //body->dump();
                //body->arg(2)->as<Global>()->init()->isa_nom<Continuation>()->dump();
                //std::cout << "TEST-->" << callee->name() << std::endl;
                found_cgra = true;
            }
        }
    });
    return found_cgra;
}


void hls_cgra_dependecy_analysis();

/**
 * @param importer hls world
 * @param Top2Kernel annonating hls_top configuration
 * @param old_world to connect with runtime (host) world
 * @return corresponding hls_top parameter for hls_launch_kernel in another world (params before rewriting kernels)
 */

//DeviceParams hls_channels(Importer& importer, Top2Kernel& top2kernel, World& old_world) {
DeviceParams hls_channels(Importer& importer, Top2Kernel& top2kernel, World& old_world, Importer& importer_cgra) {
    auto& world = importer.world(); // world is hls world
    auto& cgra_world = importer_cgra.world();
    std::vector<Def2Mode> kernels_ch_modes; // vector of channel->mode maps for kernels which use channel(s)
    std::vector<Continuation*> new_kernels;
    Def2Def kernel_new2old;
    Def2Def param2arg; // contains a map from new kernel parameters to their corresponding arguments in call-site at hls_top (for all kernels)
    Def2Def arg2param;

    std::cout << "------- OLD WORLD----------"<< std::endl;
    old_world.dump();
//    std::cout << "-------HLS----------"<< std::endl;
//    world.dump();
//
//    std::cout << "-------CGRA----------"<< std::endl;
//    cgra_world.dump();

    // hls_top should be transformed whenever there is a CGRA
    if (has_cgra_callee(old_world)) std::cout << "FOUND CGRA!" << std::endl;




//    std::vector<std::tuple<const Def*, Intrinsic>> old_world_globals;
//    for (auto def : old_world.defs()) {
//        if (auto block = def->isa_nom<Continuation>()) {
//            if (!block->has_body())
//                continue;
//            assert(block->has_body());
//            auto body = block->body();
//            auto callee = body->callee()->isa_nom<Continuation>();
//
//    auto used_globals_in_backend = [&] (Intrinsic backend) {
//            if (callee && callee->intrinsic() == backend) { //only first block of scope with be intrinsic
//                auto cont = body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
//                auto callee = cont->body()->callee()->isa_nom<Continuation>();
//                if (callee && callee->is_channel()) {
//                    auto arg = cont->body()->arg(1);
//                    if (arg->order() == 0 && !(is_mem(arg) || is_unit(arg))) {
//                        if (arg->isa_structural() && !arg->has_dep(Dep::Param)) {
//                            //cgra_global.emplace_back(arg);
//                            for (auto use : arg->uses()){
//                                //if (use->as<Continuation>())
//                                    //use->dump();
//                            }
//                            std::cout << "*************" <<std::endl;
//                            //maybe using scope.defs() to acess all bbs
//                            old_world_globals.emplace_back(arg, backend);
//                        }
//                    }
//                }
//            }
//    };
//
//        //used_globals_in_backend(Intrinsic::CGRA);
//       //used_globals_in_backend(Intrinsic::HLS);
//        }
//    }
//
//        std::cout << old_world_globals.size() <<std::endl;

    std::vector<Def2Block> old_global_maps;
    hls_cgra_global_analysis(old_world, old_global_maps);

    Def2DependentBlocks global2dependent_blocks;// [common_global, (HLS_basicblock, CGRA_basicblock)]
    hls_cgra_dependency_analysis(global2dependent_blocks, old_global_maps);



//    for (const auto& map : old_global_maps) {
//        auto map_base_addr = map.begin();
//        auto [def_global, pair] = *map_base_addr;
//        std::advance(test, 1);
//        auto test2 = (test++)->first;

    //for (size_t it = 0 ; it < old_global_maps.size();  ++it ) {
        //const auto& [current_global, current_pair] = *it;
      //    auto map =old_global_maps[it];
        //  auto def_global = map.first;
          //auto current_global = at(*it);
        //for (next_map = it + 1 ; old_global_maps.end(); ++it) 
    //for (size_t index_vector = 0; const auto& map : old_global_maps) {
      //      const auto& [global, pair] = map;
        //    for (search_index = index_vector + 1; )
          //  for (const auto& [global, pair] : map) {
            //    if (const auto& [basic_block, intrinsic] = pair ; basic_block ) {}

         //   }
//    }

    if(CheckCommon(cgra_global,hls_global))
        std::cout << "Found HLS-CGRA dependency" << std::endl;
    else
        std::cout << "No HLS-CGRA dependecy" << std::endl;






//    Scope::for_each(old_world, [&] (Scope& scope) {
//           auto kernel =  scope.entry();
//          // std::cout << "____KERNEL____" <<std::endl;
//          // kernel->dump();
//        for (auto& block : schedule(scope)) {
//            if (!block->has_body())
//                continue;
//                //block->dump();
//            assert(block->has_body());
//            auto body = block->body();
//            //auto scope_callee = scope.entry()->body()->callee();
//      //      std::cout<< "*** SCOPE *** " <<std::endl;
//      //      scope.entry()->dump();
//      //      //scope_callee->dump();
//      //      std::cout<< "*** Defs *** " <<std::endl;
//      //      for (auto def : scope.defs())
//      //          def->dump();
//                //TODO: look for read and write in defs of scopes then look at the scope then search inside all block to find that scope then look at the corresponding callee to see whether it is HLS or CGRA
//
//            auto callee = body->callee()->isa_nom<Continuation>();
//            //if (callee)
//              //  callee->dump();
//           // if (callee && callee->is_channel()) {
//           //     std::cout<< "channel" << std::endl;
//           //     std::cout << "name-->" << callee->name()<< std::endl;
//           //     body->dump();
//           // }
//            if (callee && callee->intrinsic() == Intrinsic::CGRA) {
//                //body->dump();
//            //std::cout << "init->" << std::endl;
//                //std::cout << "From CGRA->" << std::endl;
//                auto cont = body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
//                //body->arg(2)->as<Global>()->init()->isa_nom<Continuation>()->dump();
//            //std::cout << "init->cont->callee ---> " << std::endl;
//
//                //cont->body()->callee()->as_nom<Continuation>()->dump();
//                //TODO: Arg(1) is sometimes empty and it makes problem
//                //cont->body()->arg(1)->dump();
//                auto callee = cont->body()->callee()->isa_nom<Continuation>(); 
//                if (callee && callee->is_channel()) {
//                    if (cont->body()->arg(1)->order() == 0 && !(is_mem(cont->body()->arg(1)) || is_unit(cont->body()->arg(1)))) {
//                        auto def = cont->body()->arg(1);
//                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
//                            cgra_global.emplace_back(def);
//                            std::cout << "***cgra size: "  <<  cgra_global.size()<<"******" << std::endl;
//
//                        }
//                    }
//                }
//                //cont->as_nom<Continuation>();
//            }
//            if (callee && callee->intrinsic() == Intrinsic::HLS) {
//                //std::cout << "From HLS->" << std::endl;
//
//                auto cont = body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
//                //body->arg(2)->as<Global>()->init()->isa_nom<Continuation>()->dump();
//                //cont->body()->arg(1)->dump();
//                //cont->body()->dump();
//                //cont->body()->callee()->dump();
//                auto callee = cont->body()->callee()->isa_nom<Continuation>();
//                if (callee && callee->is_channel()) {
//                    if (cont->body()->arg(1)->order() == 0 && !(is_mem(cont->body()->arg(1)) || is_unit(cont->body()->arg(1)))) {
//                        auto def = cont->body()->arg(1);
//                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
//                //if (cont->body()->callee()->is_channel())
//                           // for (auto use : def->uses()) {
//                           //     //auto test = use->isa<App>();
//                           //     auto test = use->isa<App>();
//                           //     if (test) {
//                           //         auto conts = test->using_continuations();
//                           //         for (auto cont : conts)
//                           //             cont->dump();
//                           //     }
//                           // }
//                                //cont->dump(); TODO: we need to find corresponding continuations in the HLS world
//                                //importer.def_old2new_[
//                                hls_global.emplace_back(def);
//                        }
//                //body->dump();
//                //std::cout << "global pointer--> ";
//                //body->arg(2)->as<Global>()->init()->isa_nom<Continuation>()->dump();
//                //body->arg(2)->as<Global>()->init()->dump();
//                    }
//                }
//        }
//        }
//    });
//
//
//    if(CheckCommon(cgra_global,hls_global))
//        std::cout << "Found HLS-CGRA dependency" << std::endl;
//    else
//        std::cout << "No HLS-CGRA dependecy" << std::endl;
//
//



//            Def2Mode def2mode_cgra;
//    Scope::for_each(cgra_world, [&] (Scope& scope) {
//            extract_kernel_channels(schedule(scope), def2mode_cgra);
//            std::cout << "CGRA def2mode size: " <<def2mode_cgra.size() << std::endl;
//            std::cout << "CGRA is here" << std::endl;
//            });

// TODO: channels used both by CGRA and HLS  must be append to hls_top parameters

    Scope::for_each(world, [&] (Scope& scope) {
            //world.dump();
            auto old_kernel = scope.entry();
            //std::cout<< "____HLS KERNEL____" << std::endl;
            //old_kernel->dump();
            Def2Mode def2mode;
            extract_kernel_channels(schedule(scope), def2mode);
            for (auto [elem,_] : def2mode)
                std::cout << "hls: " <<  elem->unique_name()<< std::endl;

            Array<const Type*> new_param_types(def2mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());
            size_t i = old_kernel->num_params();
            // This vector records pairs containing:
            // - The position of the channel parameter for the new kernel
            // - The old global definition for the channel
            std::vector<std::pair<size_t, const Def*>> index2def;
            for (auto [def, _] : def2mode) {
                //map.first->dump();
                index2def.emplace_back(i, def);
                new_param_types[i++] = def->type();
            }

            // new kernels signature
            // fn(mem, ret_cnt, ... , /channels/ )
            auto new_kernel = world.continuation(world.fn_type(new_param_types), old_kernel->debug());
            world.make_external(new_kernel);

            kernel_new2old.emplace(new_kernel, old_kernel);

            if (is_single_kernel(new_kernel))
                new_kernels.emplace(new_kernels.begin(),new_kernel);
            else
                new_kernels.emplace_back(new_kernel);

            world.make_internal(old_kernel);

            Rewriter rewriter;
            // Map the parameters of the old kernel to the first N parameters of the new one
            // The channels used inside the kernel are mapped to the parameters N + 1, N + 2, ...
            for (auto pair : index2def) {
                auto param = new_kernel->param(pair.first);
                rewriter.old2new[pair.second] = param;
                param2arg[param] = pair.second; // (channel params, globals)
            }
            for (auto def : scope.defs()) {
                if (auto cont = def->isa_nom<Continuation>()) {
                    // Copy the basic block by calling stub
                    // Or reuse the newly created kernel copy if def is the old kernel
                    auto new_cont = def == old_kernel ? new_kernel : cont->stub();
                    rewriter.old2new[cont] = new_cont;
                    for (size_t i = 0; i < cont->num_params(); ++i)
                        rewriter.old2new[cont->param(i)] = new_cont->param(i);
                }
            }
            // Rewriting the basic blocks of the kernel using the map
            for (auto def : scope.defs()) {
                if (auto cont = def->isa_nom<Continuation>()) { // all basic blocks of the scope
                    if (!cont->has_body()) continue;
                    auto body = cont->body();
                    auto new_cont = rewriter.old2new[cont]->isa_nom<Continuation>();
                    auto new_callee = rewriter.instantiate(body->callee());
                    Array<const Def*> new_args(body->num_args());
                    for ( size_t i = 0; i < body->num_args(); ++i)
                        new_args[i] = rewriter.instantiate(body->arg(i));
                    new_cont->jump(new_callee, new_args, cont->debug());
                }
            }
            if (!is_single_kernel(new_kernel))
                kernels_ch_modes.emplace_back(def2mode);


//    for (auto [chan, mode] : def2mode) {
//            for (auto def_old : old_world.defs()) {
//                if (auto ocontinuation = def_old->isa_nom<Continuation>()) {
//                    auto ncontinuation = chan->isa_nom<Continuation>();
//                    if (auto new_ = importer.def_old2new_[ocontinuation]) {
//                        ocontinuation->dump();
//                        new_->dump();
//                        chan->as<Global>()->init()->dump();
//                    }
//                    if (ncontinuation == importer.def_old2new_[ocontinuation]) {
//                        if (ncontinuation) {
//                        //    ncontinuation->dump();
//                            std::cout << "TICK" <<std::endl;
//                        }
//                    }
//                }
//            }
//    }
    });


    // Building the type of hls_top
    std::vector<const Type*> top_param_types;
    top_param_types.emplace_back(world.mem_type());
    top_param_types.emplace_back(world.fn_type({ world.mem_type() }));
    std::vector<std::tuple<Continuation*, size_t, size_t>> param_index; // tuples made of (new_kernel, index new kernel param., index hls_top param.)
    for (auto kernel : new_kernels) {
        for (size_t i = 0; i < kernel->num_params(); ++i) {
            auto param = kernel->param(i);
            // If the parameter is not a channel, save the details and add it to the hls_top parameter list
            // TODO: if the paramete is not a channel or is a channel connected to a CGRA kernel then ...
            if (!is_channel_type(param->type())) {
                if (param != kernel->ret_param() && param != kernel->mem_param()) {
                    param_index.emplace_back(kernel, i, top_param_types.size());
                    top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    top_param_types.emplace_back(param->type());
                }
            }
        }
    }

    auto hls_top = world.continuation(world.fn_type(top_param_types), Debug("hls_top"));
    for (auto tuple : param_index) {
        // Mapping hls_top params as args for new_kernels' params
        auto param = std::get<0>(tuple)->param(std::get<1>(tuple));
        auto arg   = hls_top->param(std::get<2>(tuple));
        param2arg.emplace(param, arg); // adding (non-channel params, hls_top params as args)
        arg2param.emplace(arg, param); // channel-params are not here
    }

    // ---------- Preparing args for calling hls_top from host ------------
    // new_kernels hls world-->old_kernels in hls world-->kenels in old_world

    // Maping new_kernels params to old kernels params
    std::vector<const Def*> old_kernels_params;
    for (auto param : hls_top->params()) {
        if (arg2param.contains(param)) {
            auto new_kernel_param = arg2param[param]->as<Param>();
            auto old_kernel = kernel_new2old[new_kernel_param->continuation()];
            old_kernels_params.emplace_back(old_kernel->as_nom<Continuation>()->param(new_kernel_param->index()));
        }
    }


    // Searching in all old continuations for maping hls world params (non-channels from old kernels) to old_world params.
    for (auto& elem : old_kernels_params) {
        for (auto def : old_world.defs()) {
            if (auto ocontinuation = def->isa_nom<Continuation>()) {
                auto ncontinuation = elem->as<Param>()->continuation(); //TODO: for optimization This line can go out of inner loop
                if (ncontinuation == importer.def_old2new_[ocontinuation]) {
                    elem = ocontinuation->param(elem->as<Param>()->index());
                    break;
                }
            }
        }
    }

   // --------------------------------------------------------------



// probably need to use old2new or sending globals from cgra_graphs to here befor rewrite

    for (auto def : cgra_world.defs()) {
       // if (auto continuation = def->isa_nom<Continuation>())
       //     continuation->dump();
        //def->dump();
    }


// std::cout << "_-------------------" <<std::endl;
//    for (auto def : world.defs()) {
//       // if (auto continuation = def->isa_nom<Continuation>())
//       //     continuation->dump();
//        def->dump();
//    }

    // --------------------------------------------------------------------

    auto enter   = world.enter(hls_top->mem_param());
    auto cur_mem = world.extract(enter, 0_s);
    // hls_top memory obj frame to be used in making channel slots
    auto frame   = world.extract(enter, 1_s);

    Def2Def global2slot;
    std::vector<const Def*> channel_slots;
    std::vector<const Global*> globals;
    for (auto def : world.defs()) {
        if (auto global = def->isa<Global>()) {
           // auto cont = def->isa_nom<Continuation>();
            //if (cont)
            //    cont->dump();
            //def->dump();
            //def->as_nom<Continuation>();
            std::cout << " HLS world global_name: "<<global->unique_name() << std::endl;
            globals.emplace_back(global);
        }
    }


    for (auto def : cgra_world.defs()) {
        if (auto global = def->isa<Global>()) {
            std::cout << "CGRA world global_name: "<<global->unique_name() << std::endl;
            //globals.emplace_back(global);
        }
    }

    for (auto def : old_world.defs()) {
        if (auto global = def->isa<Global>()) {
            if (global->init()->isa<Bottom>()) { // make sure it is a only a global variable
                std::cout << "old world global_name: "<< global->unique_name() << std::endl;
                //globals.emplace_back(global);
            }
        }
    }


    Dependencies dependencies;
    // We need to iterate over globals twice because we cannot iterate over primops while creating new primops
    for (auto global : globals) {
        if (is_channel_type(global->type())) {
            channel_slots.emplace_back(world.slot(global->type()->as<PtrType>()->pointee(), frame));
            global2slot.emplace(global, channel_slots.back());
        }

        // Finding all dependencies between the kernels (building a dependency graph)
        // for each global variables find the kernels which use it,
        // check the mode on each kernel and fill a dpendency data structure: < Write, Read> => <From, To>
        // It is not possible to read a channel before writing that, so dependencies are "From write To read"
        size_t from, to = 0;
        for(size_t index_from = 0; index_from < kernels_ch_modes.size() ; ++index_from) {
          //  std::cout << "-----global----" << std::endl;
          //  global->dump();
          //  std::cout << "-----MAP----" << std::endl;
          //  for (auto [chan, _] : kernels_ch_modes[index_from])
          //      chan->dump();
            auto channel_it = kernels_ch_modes[index_from].find(global);
            if (channel_it != kernels_ch_modes[index_from].end()) {
                auto mode = channel_it->second;
                if (mode == ChannelMode::Write) {
                    from = index_from;
                    for (size_t index_to = 0; index_to < kernels_ch_modes.size(); ++index_to) {
                        auto channel_it = kernels_ch_modes[index_to].find(global);
                        if (channel_it != kernels_ch_modes[index_to].end()) {
                            auto mode = channel_it->second;
                            if (mode == ChannelMode::Read) {
                                to = index_to;
                                dependencies.emplace_back(from, to);
                            }
                        }
                    }
                }
            }
        }
     //   auto cgra_channel_it = def2mode_cgra.find(global);
     //   if (cgra_channel_it != def2mode_cgra.end()) {
     //       std::cout << "global_inside_cgra" << std::endl;
     //   }
    }

    // resolving dependencies
    std::vector<size_t> resolved;
    const size_t dependent_kernels_size = kernels_ch_modes.size();
    auto single_kernels_size = new_kernels.size() - dependent_kernels_size;
    std::vector<std::pair<size_t, size_t>> cycle;
    // Passing vector of dependencies
    if (dependency_resolver(dependencies, dependent_kernels_size, resolved)) {
        for (auto& elem : resolved)
            elem = elem + single_kernels_size;
    } else {
        world.ELOG("Kernels have circular dependency");
        // finding all circles between kernels
        for (size_t i = 0; i < dependencies.size(); ++i) {
            for (size_t j = i; j < dependencies.size(); ++j) {
                if (dependencies[i].first == dependencies[j].second &&
                        // extra condition to take into account circles in disconnected kernel networks
                        std::find(cycle.begin(), cycle.end(), dependencies[i]) == cycle.end()) {
                    cycle.emplace_back(i,j);
                }
            }
        }
        for (auto elem : cycle) {
            auto circle_from_index = dependencies[elem.first].first + single_kernels_size;
            auto circle_to_index   = dependencies[elem.second].first + single_kernels_size;
            world.ELOG("A channel between kernel#{} {} and kernel#{} {} made a circular data flow",
                    circle_from_index, new_kernels[circle_from_index]->name(),
                    circle_to_index, new_kernels[circle_to_index]->name());
        }
        assert(false && "circular dependency between kernels");
    }

    // reordering kernels, resolving dependencies
    auto copy_new_kernels = new_kernels;
    for (size_t i = 0; i < resolved.size(); ++i) {
        auto succ_kernel = resolved[i];
        new_kernels[i + single_kernels_size] = copy_new_kernels[succ_kernel];
    }

    auto cur_bb = hls_top;
    for (auto kernel : new_kernels) {
        auto ret_param = kernel->ret_param();
        auto mem_param = kernel->mem_param();
        auto ret_type = ret_param->type()->as<FnType>();
        bool last_kernel = kernel == new_kernels.back();
        const Def* hls_top_ret = hls_top->param(1);
        const Def* ret = last_kernel ? hls_top_ret : world.continuation(ret_type, Debug("next_kernel"));
        // Fill the array of arguments
        Array<const Def*> args(kernel->type()->num_ops());
        for (size_t i = 0; i < kernel->type()->num_ops(); ++i) {
            auto param = kernel->param(i);
            if (param == mem_param) {
                args[i] = cur_mem;
            } else if (param == ret_param) {
                args[i] = ret;
            } else if (auto arg = param2arg[param]) {
                args[i] = arg->isa<Global>() && is_channel_type(arg->type()) ? global2slot[arg] : arg;
            } else {
                assert(false);
            }
        }

        cur_bb->jump(kernel, args);
        if (!last_kernel) {
            auto next = ret->as_nom<Continuation>();
            cur_bb = next;
            cur_mem = next->mem_param();
        }
    }

    world.make_external(hls_top);

    debug_verify(world);
    world.cleanup();

    return old_kernels_params;
}

}
