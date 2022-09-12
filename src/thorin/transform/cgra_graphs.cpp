#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/cgra_graphs.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

void cgra_graphs(Importer& importer) {

    auto& world = importer.world();
    Def2Def kernel_new2old;
    std::vector<Continuation*> new_kernels;
    //world.dump();
    Scope::for_each(world, [&] (Scope& scope) {
        Def2Mode def2mode; // channels and their R/W modes
        extract_kernel_channels(schedule(scope), def2mode);

        auto old_kernel = scope.entry();
        Array<const Type*> new_param_types(def2mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());

            size_t channel_index = old_kernel->num_params();
            // The position of the channel parameter for the new kernel and channel defintion
            std::vector<std::pair<size_t, const Def*>> channel_param_index2def;
            for (auto [channel, _ ]: def2mode) {
                channel_param_index2def.emplace_back(channel_index, channel);
                new_param_types[channel_index++] = channel->type();
            }

            // new kernels signature
            // fn(mem, ret_cnt, ... , /channels/ )
            auto new_kernel = world.continuation(world.fn_type(new_param_types), old_kernel->debug());
            world.make_external(new_kernel);

            kernel_new2old.emplace(new_kernel, old_kernel);

            // Kernels without any channels are scheduled in the begening
            if (is_single_kernel(new_kernel))
                new_kernels.emplace(new_kernels.begin(),new_kernel);
            else
                new_kernels.emplace_back(new_kernel);

            world.make_internal(old_kernel);

            Rewriter rewriter;

    });

    world.cleanup();


}

}
