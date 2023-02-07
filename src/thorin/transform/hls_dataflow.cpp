#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

//using Def2Mode = DefMap<ChannelMode>;
using Dependencies = std::vector<std::pair<size_t, size_t>>; // (From, To)
using Def2Block    = DefMap<std::pair<Continuation*, Intrinsic>>; // [global_def , (basicblock, HLS/CGRA intrinsic)]
using Cycle        = std::vector<std::pair<size_t, size_t>>;


// makes a data structure that maps globals variables to their basic block and their HLS/ CGRA intrinsic
void hls_cgra_global_analysis(World& world, std::vector<Def2Block>& old_global_maps) {
    Scope::for_each(world, [&] (Scope& scope) {
            auto kernel = scope.entry();
            Def2Block global2block; // global, using basic block, HLS/CGRA
            for (auto& block : schedule(scope)) {
                if (!block->has_body())
                    continue;
                assert(block->has_body());

                auto body = block->body();
                auto callee = body->callee()->isa_nom<Continuation>();
                if (callee && callee->is_channel()) {
                    if (body->arg(1)->order() == 0 && !(is_mem(body->arg(1)) || is_unit(body->arg(1)))) {
                        auto def = body->arg(1);
                        if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                            for (auto preds_scope : scope.entry()->preds()) {
                                if (auto pred_scope_callee = preds_scope->body()->callee()->isa_nom<Continuation>();
                                        pred_scope_callee && pred_scope_callee->is_intrinsic()) {
                                    if (pred_scope_callee->intrinsic() == Intrinsic::HLS ||
                                        pred_scope_callee->intrinsic() == Intrinsic::CGRA) {
                                        global2block.emplace(def, std::make_pair(block, pred_scope_callee->intrinsic()));
                                    }
                                }
                            }
                        }
                    }
                }
        }

        if (!global2block.empty())
            old_global_maps.emplace_back(global2block);
    });
    }


// HLS-CGRA dependency search algorithm
// makes the main data structure that indicates how hls and cgra basic blocks get dependant by sharing common global variables.
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
                        if (cur_intrinsic == Intrinsic::CGRA)
                            std::swap(cur_basic_block, next_basic_block);
                        auto hls_basic_block  = cur_basic_block;
                        auto cgra_basic_block = next_basic_block;
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

// converts hls-cgra dependent blocks from old world to hls / cgra worlds (Target Blocks)
void connecting_blocks_old2new(std::vector<const Def*>& target_blocks, const Def2DependentBlocks def2dependent_blocks, Importer& importer, World& old_world, std::function<Continuation*(DependentBlocks)> select_block) {
    for (const auto& [old_common_global, pair] : def2dependent_blocks) {
        auto old_basicblock = select_block(pair);
                    if (importer.def_old2new_.contains(old_basicblock)) {
                        target_blocks.emplace_back(importer.def_old2new_[old_basicblock]);
            }
    }
}

void channel_mode(const Continuation* continuation, ChannelMode& mode) {
    auto app = continuation->body();
    auto callee = app->callee()->isa_nom<Continuation>();
    if (callee && callee->is_channel()) {
        if (app->arg(1)->order() == 0 && !(is_mem(app->arg(1)) || is_unit(app->arg(1)))) {
            auto def = app->arg(1);
            if (def->isa_structural() && !def->has_dep(Dep::Param)) {
                if (callee->name().find("write_channel") != std::string::npos) {
                    mode = ChannelMode::Write;
                    return;
                } else if (callee->name().find("read_channel") != std::string::npos) {
                    mode =  ChannelMode::Read;
                    return;
                } else {
                    continuation->world().ELOG("Not a channel / unsupported channel placeholder");
                    return;
                }
            }
        }
    }
}


/* HLS scheduling algorithm needs to know how hls kernels read/write from/to CGRA kernels
 we consider target CGRA kernels as if they are HLS kernels to resolve hls dependencies.
 with this aim, this function returns a vector on def2modes param, each elements of that corresponds to def2mode map of a target CGRA kernel
*/
/**
 * @param def2modes, each elements of this vector indicates a cgra kernel that connects to a hls kernel (target kernels) and includes a map betwen their globals in hls world and their read/write mode in cgra world
 * @param size_t, size of dependent blocks (target blocks size)
 * @param cgra_defs, target blocks in cgra world
 * @param hls_defs, target blocks in hls world
 */
void target_cgra_modes(std::vector<Def2Mode>& defs2modes, const size_t dependent_blocks_size, std::vector<const Def*> cgra_defs, std::vector<const Def*> hls_defs) {

    auto scope_of_block = [&] (const Continuation* cont) -> Continuation* {
        Continuation* entry_block = nullptr;
        // we look for the first inner scope entry
        auto pred_cont = cont->preds().back();
        while (pred_cont) {
            if (pred_cont->is_external()) {
                entry_block = pred_cont;
                return entry_block;
            } else {
                 pred_cont = pred_cont->preds().back();
            }
        }
        return entry_block;
    };

    // basic blocks of the same kernel are in the same map
    auto current_scope = static_cast<Continuation*>(nullptr);
    auto previous_scope = current_scope;
    Def2Mode hls_def2cgra_mode; // should be reset after each kernel (each iteration)
    for (size_t i = 0; i < dependent_blocks_size; ++i) {
        auto continuation = cgra_defs[i]->as_nom<Continuation>();
        ChannelMode mode;
        channel_mode(continuation, mode);
        // check for similar preds to find different kernels
        if (continuation->has_body() && continuation->is_returning() && continuation->is_external()) {
            // scope-entry block (external)
            auto app = continuation->body();
            current_scope = continuation;
        } else {
            // coninuation basic block, needs to find it parent, external scope-entry block
            if (auto entry_block = scope_of_block(continuation)) {
                current_scope = entry_block;
            }
        }

        auto hls_app = hls_defs[i]->as_nom<Continuation>()->body();
        auto callee = hls_app->callee()->isa_nom<Continuation>();
        if (callee && callee->is_channel()) {
            if (hls_app->arg(1)->order() == 0 && !(is_mem(hls_app->arg(1)) || is_unit(hls_app->arg(1)))) {
                auto def = hls_app->arg(1); // this def is a global
                hls_def2cgra_mode.emplace(def, mode);
            }
        }

        // first iteration
        if (previous_scope == nullptr) {
            previous_scope = current_scope;
            continue;
        }

        // last iteration and no new kernel, save the only map
        if (current_scope == previous_scope && i == dependent_blocks_size - 1) {
            defs2modes.emplace_back(hls_def2cgra_mode);
            hls_def2cgra_mode.clear();
            return;
        }

        if (current_scope != previous_scope) {
            // it is a new kernel save the map and clear it
            defs2modes.emplace_back(hls_def2cgra_mode);
            hls_def2cgra_mode.clear();
        }

        previous_scope = current_scope;
    }
   // defs2modes.size() is equal to the number of CGRA kernels that connect to hls

}


inline void extract_kernel_channels(const Continuation* continuation, Def2Mode& def2mode) {

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

void extract_kernel_channels(const Schedule& schedule, Def2Mode& def2mode) {
    for (const auto& continuation : schedule) {
        if (!continuation->has_body())
            continue;
        extract_kernel_channels(continuation, def2mode);
    }
}

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
    // adding hls_top param sizes to configuraion
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
            if (callee && callee->intrinsic() == Intrinsic::CGRA) {
                found_cgra = true;
            }
        }
    });
    return found_cgra;
}

void circle_analysis(Dependencies dependencies, World& world, size_t single_kernels_size, std::vector<Continuation*> new_kernels) {

    Cycle cycle;
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

    auto has_cgra_circle = [&] (const auto& elem) {
        return dependencies[elem.first].first + single_kernels_size  >= new_kernels.size() ||
            dependencies[elem.second].first + single_kernels_size >= new_kernels.size();
    };

    auto has_hls_circle = [&] (const auto& elem) {
        return dependencies[elem.first].first + single_kernels_size  < new_kernels.size() ||
            dependencies[elem.second].first + single_kernels_size < new_kernels.size();
    };

    auto cgra_circle = false;
    if (std::any_of(cycle.cbegin(), cycle.cend(), has_cgra_circle)) {
        world.ELOG("CGRA kernel(s) inside the circle");
        cgra_circle = true;
    }
    auto hls_circle = false;
    if (std::any_of(cycle.cbegin(), cycle.cend(), has_hls_circle)) {
        world.ELOG("HLS kernel(s) inside the circle");
        hls_circle = true;
    }

    auto circle_from_index = 0, circle_to_index = 0;
    for (auto elem : cycle) {
        circle_from_index = dependencies[elem.first].first + single_kernels_size;
        circle_to_index   = dependencies[elem.second].first + single_kernels_size;
        if (!cgra_circle) {
            world.ELOG("A channel between HLS kernel#{} {} and HLS kernel#{} {} made a circular data flow",
                    circle_from_index, new_kernels[circle_from_index]->name(),
                    circle_to_index, new_kernels[circle_to_index]->name());
        } else if(cgra_circle && hls_circle) { // find only hls index and show the kernel
            if (circle_from_index < new_kernels.size() && circle_from_index < new_kernels.size()) {
                world.ELOG("A channel between HLS kernel#{} {} and HLS kernel#{} {} made a circular data flow",
                        circle_from_index, new_kernels[circle_from_index]->name(),
                        circle_to_index, new_kernels[circle_to_index]->name());
            }
        }
    }
    if (!cgra_circle)
        assert(false && "circular dependency between HLS kernels is not supported!");
}


/**
 * @param importer hls world
 * @param Top2Kernel annonating hls_top configuration
 * @param old_world to connect with runtime (host) world
 * @return corresponding hls_top parameter for hls_launch_kernel in another world (params before rewriting kernels)
 */

DeviceDefs hls_dataflow(Importer& importer, Top2Kernel& top2kernel, World& old_world, Importer& importer_cgra) {
    auto& world = importer.world(); // world is hls world
    auto& cgra_world = importer_cgra.world();
    // TODO: rename to hls_new_kernels
    // the size of this vector is equal to the size of kernels with deps.
    // we need a similar size for cgra. i.e the size of kernels with deps.
    std::vector<Def2Mode> kernels_ch_modes; // vector of channel->mode maps for kernels which use channel(s)
    std::vector<Continuation*> new_kernels;
    Def2Def kernel_new2old;
    Def2Def param2arg; // contains a map from new kernel parameters to their corresponding arguments in call-site at hls_top (for all kernels)
    Def2Def arg2param;

    // hls_top should be transformed whenever there is a CGRA
    if (has_cgra_callee(old_world)) std::cout << "FOUND CGRA!" << std::endl;


    std::vector<Def2Block> old_global_maps;
    hls_cgra_global_analysis(old_world, old_global_maps);

    Def2DependentBlocks old_globals2old_dependent_blocks;// [common_global, (HLS_basicblock, CGRA_basicblock)]
    hls_cgra_dependency_analysis(old_globals2old_dependent_blocks, old_global_maps);
    old_global_maps.clear();


    std::vector<const Def*> target_blocks_in_hls_world; // hls_world basic blocks that connect to CGRA`
    connecting_blocks_old2new(target_blocks_in_hls_world, old_globals2old_dependent_blocks, importer, old_world, [&] (DependentBlocks dependent_blocks) {
        auto old_hls_basicblock = dependent_blocks.first;
        return old_hls_basicblock;
    });

    Scope::for_each(world, [&] (Scope& scope) {
            auto old_kernel = scope.entry();
            // def is a global in hls world
            // mode states the global is written/read in a kernel
            Def2Mode hls_def2hls_mode;
            extract_kernel_channels(schedule(scope), hls_def2hls_mode);

            Array<const Type*> new_param_types(hls_def2hls_mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());
            size_t i = old_kernel->num_params();
            // This vector records pairs containing:
            // - The position of the channel parameter for the new kernel
            // - The old global definition for the channel
            std::vector<std::pair<size_t, const Def*>> index2def;
            for (auto [def, _] : hls_def2hls_mode) {
                // num_param can be used as index directly since index always starts from zero
                index2def.emplace_back(i, def);
                new_param_types[i++] = def->type();
            }

            // new kernels signature
            // fn(mem, ret_cnt, ... , /channels/ )
            auto new_kernel = world.continuation(world.fn_type(new_param_types), old_kernel->debug());
            world.make_external(new_kernel);

            kernel_new2old.emplace(new_kernel, old_kernel);

            if (is_single_kernel(new_kernel))
                new_kernels.emplace(new_kernels.begin(), new_kernel);
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
                        rewriter.old2new[cont->param(i)] = new_cont->param(i); //non-channel params
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
                kernels_ch_modes.emplace_back(hls_def2hls_mode);
    });


    // Building the hls_top kernel
    std::vector<const Type*> top_param_types;
    top_param_types.emplace_back(world.mem_type());
    top_param_types.emplace_back(world.fn_type({ world.mem_type() }));
    std::vector<std::tuple<Continuation*, size_t, size_t>> param_index; // tuples made of (new_kernel, index new kernel param, index hls_top param.)

    // We check for the corresponding globals that channel-params are mapped to
    // then we look for all using basic blocks and check if they are among the blocks that are connected to CGRA
    // note that in each basic block only one unique global can be read or written
    auto is_used_for_cgra = [&] (const Def* param) -> bool  {
        if (is_channel_type(param->type())) {
            if (auto global = param2arg[param]; !global->empty()) {// at this point only (channel params, globals) are available inside the map
                for (auto use : global->uses()) {
                    if (auto app = use->isa<App>()) {
                        auto ucontinuations = app->using_continuations();
                        for (const auto& block : target_blocks_in_hls_world) {
                            if (std::find(ucontinuations.begin(), ucontinuations.end(), block) != ucontinuations.end())
                                return true;
                            }
                        }
                    }
                }
            }
        return false;
    };

    std::vector<const Def*> target_blocks_in_cgra_world; // cgra_world basic blocks that connect to hls
    connecting_blocks_old2new(target_blocks_in_cgra_world, old_globals2old_dependent_blocks, importer_cgra, old_world, [&] (DependentBlocks dependent_blocks) {
        auto old_cgra_basicblock = dependent_blocks.second;
        return old_cgra_basicblock;
    });

    for (auto kernel : new_kernels) {
        for (size_t i = 0; i < kernel->num_params(); ++i) {
            auto param = kernel->param(i);
            // If the parameter is not a channel or is a channel but connected to a CGRA kernel then
            // save the index and add it to the hls_top parameter list
            if (!is_channel_type(param->type())) {
                if (param != kernel->ret_param() && param != kernel->mem_param()) {
                    param_index.emplace_back(kernel, i, top_param_types.size());
                    top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    top_param_types.emplace_back(param->type());
                }
            } else if (is_used_for_cgra(param)) {
                    param_index.emplace_back(kernel, i, top_param_types.size());
                    // cgra_channels on hls_top are scalar vars, probably we don't need to add them to top2kernel
                    //top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    top_param_types.emplace_back(param->type());
                }
            }
        }

    auto hls_top = world.continuation(world.fn_type(top_param_types), Debug("hls_top"));

    for (auto tuple : param_index) {
        // Mapping hls_top params as args for new_kernels' params
        auto param = std::get<0>(tuple)->param(std::get<1>(tuple));
        auto arg   = hls_top->param(std::get<2>(tuple));
        if (is_used_for_cgra(param)) {
            // updating the args of the the already inserted channel-params
            // note that emplace method only adds a new (keys,value) and does not update/rewrite values for already inserted keys
            param2arg[param] = arg;
          //  if (param2arg.contains(param)) {
          //      std::cout << "This param is already in the map" << std::endl;
          //  }
                continue;
        }
        param2arg.emplace(param, arg); // adding (non-channel params, hls_top params as args). Channel params were added before
        arg2param.emplace(arg, param); // channel-params are not here.
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


    auto enter   = world.enter(hls_top->mem_param());
    auto cur_mem = world.extract(enter, 0_s);
    // hls_top memory obj frame to be used in making channel slots
    auto frame   = world.extract(enter, 1_s);

    Def2Def global2slot;
    std::vector<const Def*> channel_slots;
    std::vector<const Global*> globals;
    for (auto def : world.defs()) {
        if (auto global = def->isa<Global>()) {
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
//    std::cout << "--------HLS after rewrite -----" << std::endl;
//    world.dump();
    world.cleanup();

    return std::make_tuple(old_kernels_params, old_globals2old_dependent_blocks);
}

}
