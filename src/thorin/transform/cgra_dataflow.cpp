#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/cgra_dataflow.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

PortIndices external_ports_index(const Def2Def global2param, Def2Def param2arg, const Def2DependentBlocks def2dependent_blocks, Importer& importer) {
    Array<size_t> param_indices(def2dependent_blocks.size());
    size_t i = 0;
    for (auto it = def2dependent_blocks.begin(); it != def2dependent_blocks.end(); ++it) {
        auto old_common_global = it->first; //def type
        if (importer.def_old2new_.contains(old_common_global)) {
            for (const auto& [global, param] : global2param) {
                if (global == importer.def_old2new_[old_common_global]) {
                    // this param2arg is after replacing global args with hls_top params that connect to cgra
                    // basically we can name it kernelparam2hls_top_cgra_param
                    auto top_param = param2arg[param];
                    param_indices[i++] = top_param->as<Param>()->index();
                }
            }

        }
    }
    return param_indices;
}


Array<size_t> cgra_dataflow(Importer& importer, World& old_world, Def2DependentBlocks& def2dependent_blocks) {

    auto& world = importer.world();
// std::cout << "_--------cgra world before rewrite--------" <<std::endl;
//    world.dump();
    std::vector<const Def*> target_blocks_in_cgra_world; // cgra_world basic blocks that connect to HLS
    connecting_blocks_old2new(target_blocks_in_cgra_world, def2dependent_blocks, importer, [&] (DependentBlocks dependent_blocks) {
        auto old_cgra_basicblock = dependent_blocks.second;
        return old_cgra_basicblock;
    });

    //Def2Def kernel_new2old;
    std::vector<Continuation*> new_kernels;
    Def2Def param2arg; // contains map from new kernel channel-parameters to channels (globals)

    Scope::for_each(world, [&] (Scope& scope) {
        Def2Mode def2mode; // channels and their R/W modes
        extract_kernel_channels(schedule(scope), def2mode);

        auto old_kernel = scope.entry();
        // for each kernel new_param_types contains both the type of kernel parameters and the channels used inside that kernel
        Array<const Type*> new_param_types(def2mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());

            size_t channel_index = old_kernel->num_params();
            // The position of the channel parameters in new kernels and their corresponding channel defintion
            std::vector<std::pair<size_t, const Def*>> channel_param_index2def;
            for (auto [channel, _ ]: def2mode) {
                channel_param_index2def.emplace_back(channel_index, channel);
                new_param_types[channel_index++] = channel->type();
            }

            // new kernels signature
            // fn(mem, ret_cnt, ... , /channels/ )
            auto new_kernel = world.continuation(world.fn_type(new_param_types), old_kernel->debug());
            world.make_external(new_kernel);

            //kernel_new2old.emplace(new_kernel, old_kernel);

            // Kernels without any channels are scheduled in the begening
            if (is_single_kernel(new_kernel))
                new_kernels.emplace(new_kernels.begin(),new_kernel);
            else
                new_kernels.emplace_back(new_kernel);

            world.make_internal(old_kernel);

            Rewriter rewriter;

          // rewriting channel parameters
            for (auto [channel_param_index, channel] : channel_param_index2def) {
                auto channel_param = new_kernel->param(channel_param_index);
                rewriter.old2new[channel] = channel_param;
                // In CGRA ADF only connected nodes (params) are concerned and there is no need to
                // introduce a new variable (like mem slots for channels) to connect them together.
                // so at this point param2arg map is not required.
                //  param2arg[channel_param] = channel; // (channel as kernel param, channel as global)
                  param2arg[channel_param] = channel; // (channel as kernel param, channel as global)
            }

          // rewriting basicblocks and their parameters
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
            // The rewrite eventually maps the parameters of the old kernel to the first N parameters of the new one
            // The channels used inside the kernel are mapped to the parameters N + 1, N + 2, ...
            for (auto def : scope.defs()) {
                if (auto cont = def->isa_nom<Continuation>()) { // all basic blocks of the scope
                    if (!cont->has_body()) continue;
                    auto body = cont->body();
                    auto new_callee = rewriter.instantiate(body->callee());

                    Array<const Def*> new_args(body->num_args());
                    for ( size_t i = 0; i < body->num_args(); ++i)
                        new_args[i] = rewriter.instantiate(body->arg(i));

                    auto new_cont = rewriter.old2new[cont]->isa_nom<Continuation>();
                    new_cont->jump(new_callee, new_args, cont->debug());
                }
            }
    });


//    std::cout << "Target block size = " << target_blocks_in_cgra_world.size() << std::endl; 
//    for (const auto& block : target_blocks_in_cgra_world) {
//        std::cout << "Target block"  << std::endl;
//        block->dump();
//    }
    // We check for the corresponding globals that channel-params are mapped to
    // then we look for all using basic blocks and check if they are among the blocks that are connected to HLS
    // note that in each basic block only one unique global can be read or written
    auto is_used_for_hls = [&] (const Def* param) -> bool  {
    if (is_channel_type(param->type())) {
        if (auto global = param2arg[param]; !global->empty()) {// at this point only (channel params, globals) are available inside the map
            for (auto use : global->uses()) {
                if (auto app = use->isa<App>()) {
                    auto ucontinuations = app->using_continuations();
                    for (const auto& block : target_blocks_in_cgra_world) {
                        if (std::find(ucontinuations.begin(), ucontinuations.end(), block) != ucontinuations.end())
                            return true;
                        }
                    }
                }
            }
        }
    return false;
    };

    // finding the ports (channels connected to HLS kernels)
    for (auto kernel : new_kernels) {
        for (size_t i = 0; i < kernel->num_params(); ++i) {
            auto param = kernel->param(i);
            // If the parameter is not a channel or is a channel but connected to a CGRA kernel then
            // save the index and add it to the hls_top parameter list
            if (!is_channel_type(param->type())) {
                if (param != kernel->ret_param() && param != kernel->mem_param()) {
              //      param_index.emplace_back(kernel, i, top_param_types.size());
              //      top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
              //      top_param_types.emplace_back(param->type());
                }
            } else if (is_used_for_hls(param)) {
                    //param_index.emplace_back(kernel, i, top_param_types.size());
                    //top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    //top_param_types.emplace_back(param->type());
                    std::cout << "A PORT found!" << std::endl;
                }
            }
        }


    std::vector<const Type*> graph_param_types;
    graph_param_types.emplace_back(world.mem_type());
    graph_param_types.emplace_back(world.fn_type({ world.mem_type() }));
    auto cgra_graph = world.continuation(world.fn_type(graph_param_types), Debug("cgra_graph"));
    // jump similar to hls_top
    world.make_external(cgra_graph);

 std::cout << "_--------cgra world After rewrite--------" <<std::endl;
    world.dump();

    world.cleanup();
}

}
// check to which global a param is mapped.
// or find the corresponding param which a global is mapped to.
