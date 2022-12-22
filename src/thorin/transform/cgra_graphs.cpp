#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/cgra_graphs.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {


void cgra_graphs(Importer& importer, World& old_world, Def2DependentBlocks& def2dependent_blocks) {

    auto& world = importer.world();
    std::vector<const Def*> target_blocks_in_cgra_world; // cgra_world basic blocks that connect to HLS
    connecting_blocks_old2new(target_blocks_in_cgra_world, def2dependent_blocks, importer, old_world, [&] (DependentBlocks dependent_blocks) {
        auto old_cgra_basicblock = dependent_blocks.second;
        return old_cgra_basicblock;
    });

// std::cout << "_--------cgra before rewrite--------" <<std::endl;
//    for (auto def : world.defs()) {
       // if (auto continuation = def->isa_nom<Continuation>())
       //     continuation->dump();
       // def->dump();
//    }
    //Def2Def kernel_new2old;
    std::vector<Continuation*> new_kernels;
    //Def2Def param2arg; // contains map from new kernel parameter to arguments of calls (globals)

    Scope::for_each(world, [&] (Scope& scope) {
        Def2Mode def2mode; // channels and their R/W modes
        extract_kernel_channels(schedule(scope), def2mode);
        //world.dump();

            for (auto [elem,_] : def2mode) {
                std::cout << "cgra: " <<  elem->unique_name()<< std::endl;

            }

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

    //world.dump();
    world.cleanup();
}

}
