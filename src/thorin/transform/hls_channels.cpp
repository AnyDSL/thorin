#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/util/log.h"
#include "thorin/type.h"

namespace thorin {

enum class ChannelMode : uint8_t {
    Read,       ///< Read-channel
    Write       ///< Write-channe
};

typedef DefMap<ChannelMode> Def2Mode;

Def2Mode  def2mode_; //channel-types to channel-modes
std::vector<const Def*> top_io_params;

static void extract_kernel_channels(const Schedule& schedule) {
    for (const auto& block : schedule) {
        auto continuation = block.continuation();
        if (continuation->empty())
            continue;
        auto callee = continuation->callee()->isa_continuation();
        if (callee && callee->is_channel()) {
            if (continuation->arg(1)->order() == 0 && !(is_mem(continuation->arg(1)) || is_unit(continuation->arg(1)))) {
                auto def= continuation->arg(1);
                if (def->isa<PrimOp>() && is_const(def)) {
                    if (callee->name().str().find("write_channel") != std::string::npos) {
                        assert((!def2mode_.contains(def) || def2mode_[def] == ChannelMode::Write) &&
                                "Duplicated channel or \"READ\" mode channel redefined as WRITE!");
                        def2mode_.emplace(def,ChannelMode::Write);
                    } else if (callee->name().str().find("read_channel") != std::string::npos) {
                        assert((!def2mode_.contains(def) || def2mode_[def] == ChannelMode::Read)  &&
                                "Duplicated channel or \"WRITE\" mode channel redefined as READ!");
                        def2mode_.emplace(def,ChannelMode::Read);
                    } else {
                        ELOG("Not a channel / unsupported channel placeholder");
                    }
                }
            }
        }
    }
}


void hls_channels(World& world) {
    std::vector<Def2Mode> channels_map; // vector of channel_stmt->mode maps for each kernel

    Scope::for_each(world, [&] (Scope& scope) {
        auto kernel = scope.entry();
        extract_kernel_channels(schedule(scope));

        Array<const Type*> new_param_types(def2mode_.size() + kernel->num_params());
        std::copy(kernel->type()->ops().begin(),
                  kernel->type()->ops().end(),
                  new_param_types.begin());
        size_t i = kernel->num_params();
        std::vector<std::pair<size_t, const Def*>> index2def;
        for (auto map : def2mode_) {
            index2def.emplace_back(i, map.first); // [index, channels(formly globals)] pairs for each kernel
            new_param_types[i++] = map.first->type();
        }
        auto new_kernel = world.continuation(world.fn_type(new_param_types), kernel->debug());
        new_kernel->make_external();
        kernel->make_internal();

        Rewriter rewriter;
        // filling old2new (def2def) map with new kernel params, cont, and their params
        // (preparing rewrite rules for rewriter)
        for (auto pair : index2def)
            rewriter.old2new[pair.second] = new_kernel->param(pair.first);
        for (auto def : scope.defs()) {
            if (auto cont = def->isa_continuation()) {
                auto new_cont = def == kernel ? new_kernel : cont->stub(); //stub=> conts of new_kernel
                rewriter.old2new[cont] = new_cont;
                for (size_t i = 0; i < cont->num_params(); ++i)
                    rewriter.old2new[cont->param(i)] = new_cont->param(i);
            }
        }
        // Getting an instance of rewritten data and jump
        // each cont have a callee and a set of ops (here args), we should jump to new callees with new args
        for (auto def : scope.defs()) {
            if (auto cont = def->isa_continuation()) { // all continuations
                auto new_cont = rewriter.old2new[cont]->as_continuation();
                auto new_callee = rewriter.instantiate(cont->callee());
                Array<const Def*> new_args(cont->num_args());
                for ( size_t i = 0; i < cont->num_args(); ++i)
                    new_args[i] = rewriter.instantiate(cont->arg(i));
                new_cont->jump(new_callee, new_args, cont->debug());
            }
        }

        // Top module io params retrieval
        for (auto param : new_kernel->params()) {
            auto ptr_type = param->type()->isa<PtrType>();
            if (ptr_type && ptr_type->pointee()->isa<ArrayType>()) {
                top_io_params.emplace_back(param);
            }
        }

//        for(auto map : def2mode_) {
//            if (map.second == ChannelMode::Read ){
//                channels.emplace_back(map.first);
//            }
//        }

        channels_map.emplace_back(def2mode_);
        def2mode_.clear();
    });

    if (top_io_params.size() > 2)
        WLOG("Top function has several input/output streams");

    // Building Top types
    Array<const Type*> top_param_types(top_io_params.size() + 2);
    top_param_types[0] = world.mem_type();                      //mem
    top_param_types[1] = world.fn_type({ world.mem_type() });   //ret_cont
    for (size_t i=0; i < top_io_params.size(); ++i)
      top_param_types[i + 2] = top_io_params[i]->type();

    auto hls_top = world.continuation(world.fn_type(top_param_types), Debug("hls_top"));

    // Write a loop to retrieve the types of all different channels
    //
    auto channel_type = channels_map[1].begin()->first->type();
    std::cout<< "kernel_count = " << channels_map.size() <<endl;

    auto channel_cnt = 0;


    // Making memory slots
    auto mem_info = world.enter(hls_top->mem_param());
    auto mem_obj = world.extract(mem_info, 0_s);
    auto frame_ = world.extract(mem_info, 1_s);
    for (auto kernel : channels_map) {
        for (auto channel : kernel) {
            // channels are either in R or W mode in each kernel
            if (channel.second == ChannelMode::Write) {
                channel_cnt++;
                channel.first->type()->as<PtrType>()->pointee()->dump();
            }
        }
    }
    std::cout<<"channel_count = " << channel_cnt <<endl;
    auto slot = world.slot(channel_type->as<PtrType>()->pointee(), frame_);

    //dummy_val
    auto val = world.literal_qu8(20, {});

    auto cur_mem = world.store(mem_obj, slot, val, Debug("twenty"));
    //last kernel should jump to top return
    hls_top->jump(hls_top->ret_param(), {cur_mem});
    //hls_top->jump(hls_top->ret_param(), {cur_mem});
    //hls_top->jump(hls_top->ret_param(), { hls_top->mem_param() });
    //Call hls_top_;
    //auto target = drop(hls_top_);
    //hls_top->jump(target->callee(), target->args());

//    for (auto param : hls_top->params()) {
//        param->type()->dump();
//    }
    hls_top->make_external();


    world.dump();
    channels_map.clear();
    world.cleanup();
}


}
