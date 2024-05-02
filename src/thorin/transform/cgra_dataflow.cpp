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


void detect_param_modes(const Continuation* continuation, Def2Mode& def2mode){
    if (continuation->is_cgra_graph()) {
        continuation->num_params();
    }

//TODO: for non-channel params check R if there is a load (after lea) or a W using stor
// but a param used in any primop can also be considered a R
// a param used only in store is W
// a param used both in PrimOP and store is RW

}


void detect_param_modes(const Schedule& schedule, Def2Mode& def2mode) {
    for (const auto& continuation : schedule) {
        if (!continuation->has_body())
            continue;
        detect_param_modes(continuation, def2mode);
    }
}

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

// This annotation applies for all kernels except for cgra_graph
void annotate_channel_modes(const Continuation* imported, const ContName2ParamModes cont2param_modes, CGRAKernelConfig::Param2Mode& param2mode) {
    // The order that channel modes are inserted in param_modes cosecuteviley is aligned with the order that channels appear in imported continuations
    // for example, the first mode in param_modes (index = 0) is equal to the first channel in the imported continuation (kernel)
    for (auto const& [cont_name, param_modes] : cont2param_modes) {
        if (cont_name == imported->name()) {
            size_t index = 0;
            for (auto const& param : imported->params()) {
                if ((param->index() < 2) || is_mem(param) || param->order() != 0 || is_unit(param))
                    continue;
                else if (auto type = param->type(); is_channel_type(type)) {
                    param2mode.emplace(param, param_modes[index++]);
                }

            }

                break;
        }
    }
}

void annotate_interface(Continuation* imported, const Continuation* use) {
// the better way to avoid name comparision is using def_ol2new_ and finding the exact cont
// consider that such a function should be called after clean up.
    if (use->has_body()) {
        auto ubody = use->body();
        auto ucallee = ubody->callee();
        if (auto ucontinuation = ucallee->isa_nom<Continuation>()) {
            if (ucontinuation->intrinsic() == Intrinsic::CGRA) {
                auto ukernel = ubody->arg(5)->as<Global>()->init();
                auto ikernel = imported;
                if (ikernel->name() == ukernel->unique_name()) {
                    std::cout << "MOUSE FOUND!" << std::endl;
                    auto uiterface = ucallee->as_nom<Continuation>()->get_interface();
                    auto ubuf_size = ucallee->as_nom<Continuation>()->get_buf_size();
                    // annotating (modifying interface attribute) of enrty block of imported continuation
                    ikernel->set_interface(uiterface);
                    ikernel->set_buf_size(ubuf_size);
                    // extending the annotation to all blocks of the imported continuation
                    // this solution works and for interface implementation (continuation-wise and not param-wise) makes the
                    // c-backend simpler
                    // TODO: default to stream
                    Scope scope(ikernel);
                    for (auto& block : schedule(scope)) {
                        block->set_interface(uiterface);
                        block->set_buf_size(ubuf_size);
                    }
                }
            }
        }
    }
    return;
}

// This annotation applies only for cgra_graph
// TODO: probably we can add param sizes here and rename thid function to annotate_cgra_graph.
void annotate_cgra_graph_modes(Continuation* continuation, const Ports& hls_cgra_ports, Cont2Config& cont2config) {
    //TODO: At the moment cgra ports' modes are determined using the duals of HLS ports which is probably not a proper solution
    // using index2mode similar to hls is a more general solution
    //assert(hls_cgra_ports.size() > 0 && "No HLS-CGRA ports to annotate");
    CGRAKernelConfig::Param2Mode param2mode;
    if (!hls_cgra_ports.empty()) {
        //Cont2Config cont2config;
        for (const auto& ports : hls_cgra_ports) {
            auto [hls_param2mode, cgra_param] = ports;
            auto [hls_param, hls_param_mode] = hls_param2mode.value();
            //TODO: use external for cgra cont2config, use hls mode and cgra_param of ports for cgra param2mode
            // use hls_param2mode only after get_port then we can get cgra cont from param also
            // putting into config either here or inside get_kernel_configs
            ChannelMode cgra_param_mode;
            if (hls_param_mode == ChannelMode::Write)
                cgra_param_mode = ChannelMode::Read;
            else if (hls_param_mode == ChannelMode::Read)
                cgra_param_mode = ChannelMode::Write;
            else
                cgra_param_mode = ChannelMode::Undef;
            //cgra_param.value()->dump();
            //cgra_param.value()->as<Param>()->continuation()->dump();

            param2mode.emplace(cgra_param.value(), cgra_param_mode);

        }

    }
    //TODO: this way of getting cgra_graph cont only works when we have channel params
    //auto cgra_graph_cont = std::begin(param2mode)->first->as<Param>()->continuation();
    auto cgra_graph_cont = continuation;
    //TODO: At the moment  we assume all non-channel params have Undef mode
    for (const auto& param : cgra_graph_cont->params()) {
        if ((param->index() < 2) || is_mem(param) || param->order() != 0 || is_unit(param))
            continue;
            if (!is_channel_type(param->type()))
                param2mode.emplace(param, ParamMode::Undef);
    }
    for (auto [param, mode] : param2mode) {
        std::cout << "PARAM: ";
        param->dump();
        std::cout << "MODE: ";
        if (mode == ChannelMode::Read) {std::cout << "Read"<< std::endl;
        } else if (mode == ChannelMode::Write) {
            std::cout << "Write" << std::endl;
        } else if (mode == ChannelMode::Undef) {
            std::cout << "Undef" << std::endl;
        }
    }
    cont2config.emplace(cgra_graph_cont, std::make_unique<CGRAKernelConfig>(-1 , std::pair{-1,-1}, -1, param2mode, false));
}



//void annotate_cgra_graph_param_size(Continuation* continuation, const Ports& hls_cgra_ports, Cont2Config& cont2config) {};



//Array<size_t> cgra_dataflow(Importer& importer, World& old_world, Def2DependentBlocks& def2dependent_blocks) {
CgraDeviceDefs cgra_dataflow(Importer& importer, World& old_world, Def2DependentBlocks& def2dependent_blocks) {

    auto& world = importer.world();
// std::cout << "_--------cgra world before rewrite--------" <<std::endl;
//    world.dump();


    for (auto [_, cont] : old_world.externals()) {
        Scope scope(cont);
        for (auto& block : schedule(scope)) {
            if (!block->has_body())
                continue;
            assert(block->has_body());
            auto body = block->body();
            auto callee = body->callee()->isa_nom<Continuation>();
            if (callee && callee->is_intrinsic())
                if (callee->intrinsic() == Intrinsic::CGRA)
                    if (callee->get_interface() == Interface::Stream) {
                        std::cout << "STREAM INTERFACE FOUND!" << std::endl;
                        //TODO we need use old2new on already available cgra_world
                    }
        }
    }


    std::vector<const Def*> target_blocks_in_cgra_world; // cgra_world basic blocks that connect to HLS
    connecting_blocks_old2new(target_blocks_in_cgra_world, def2dependent_blocks, importer, [&] (DependentBlocks dependent_blocks) {
        auto old_cgra_basicblock = dependent_blocks.second;
        return old_cgra_basicblock;
    });

    //Def2Def kernel_new2old;
    std::vector<Continuation*> new_kernels;
    Def2Def param2arg; // contains map from new kernel channel-parameters to channels (globals)
    ContName2ParamModes kernel_name2chan_param_modes; // contains map from new kernel to its channel parameter modes
    Scope::for_each(world, [&] (Scope& scope) {
        Def2Mode def2mode; // channels and their R/W modes
        extract_kernel_channels(schedule(scope), def2mode);



        auto old_kernel = scope.entry();
        //old_kernel->set_interface();
        // for each kernel new_param_types contains both the type of kernel parameters and the channels used inside that kernel
        Array<const Type*> new_param_types(def2mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());

            size_t channel_index = old_kernel->num_params();

            // The position of the channel parameters in new kernels and their corresponding channel defintion
            Array<ParamMode> modes(def2mode.size());
            size_t i = 0;
            std::vector<std::pair<size_t, const Def*>> channel_param_index2def;
            for (auto [channel, mode]: def2mode) {
                modes[i++] = mode;
                channel_param_index2def.emplace_back(channel_index, channel);
                new_param_types[channel_index++] = channel->type();
            }

            // new kernels signature
            // fn(mem, ret_cnt, ... , /channels/ )
            //auto new_kernel = world.continuation(world.fn_type(new_param_types), Interface::Stream, old_kernel->debug());
            auto new_kernel = world.continuation(world.fn_type(new_param_types), old_kernel->debug());
            world.make_external(new_kernel);
            //new_kernel->set_interface();


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
                param2arg[channel_param] = channel; // (channel as kernel param, channel as global)
            }

          // rewriting basicblocks and their parameters
            for (auto def : scope.defs()) {
                if (auto cont = def->isa_nom<Continuation>()) {
                    // Copy the basic block by calling stub
                    // Or reuse the newly created kernel copy if def is the old kernel
                    auto new_cont = def == old_kernel ? new_kernel : cont->stub();
                    //new_cont->set_interface();
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
                    //new_cont->set_interface();
                    new_cont->jump(new_callee, new_args, cont->debug());
                    //new_cont->set_interface();
                }
            }

    kernel_name2chan_param_modes.emplace_back(new_kernel->name(), modes);

    // lowering interface attr from old world to cgra kernels
    // trying by externals , if didn't work try by defs. 


//TODO: try withouth rewriter and just jump
// use kernel config
    for (auto def: old_world.defs()) {
        if (auto ocontinuation = def->isa_nom<Continuation>()) {
            if (ocontinuation->has_body()) {
                auto body = ocontinuation->body();
                if (auto callee = body->callee()->isa_nom<Continuation>()) {
                    if (callee->is_intrinsic()) {
                        if (callee->intrinsic() == Intrinsic::CGRA) {

                            if (ocontinuation->get_interface() == Interface::Stream) {
                                std::cout << "OLD ASS STREAM1" << std::endl;
                            }
                            //if (callee->get_interface() == Interface::Stream) {
                            if (callee->get_interface() == Interface::Stream) {
                                // TODO: we probably need set interface function in thorin
                                // OR maybe we need to rewrite!
                                // NO! we cannot use NAME because attr is different from CONT, check set attr in throin
                                std::cout << "OLD ASS STREAM2" << std::endl;
                            }
                            if (auto okernel = body->arg(5)->as<Global>()->init()) {
                                auto nkernel = importer.def_old2new_[okernel];
                                auto nkernel_cont = nkernel->as_nom<Continuation>();
                                //nkernel_cont->attributes().interface = callee->interface();
                                //TODO: IT WORKS here!
                                //nkernel_cont->set_interface();
                                //auto new_cont = world.continuation(world.fn_type(nkernel_cont->type()->ops()), callee->interface(), nkernel_cont->debug());
                                //auto new_cont = world.continuation(world.fn_type(nkernel_cont->type()->ops()), Interface::Stream, nkernel_cont->debug());
                                //auto new_cont = world.continuation(world.fn_type(nkernel_cont->type()->ops()), nkernel_cont->debug());


                              //  auto body = nkernel_cont->body();
                              //  auto new_cont_ = rewriter.old2new[nkernel_cont]->isa_nom<Continuation>();
                              //  auto new_callee = rewriter.instantiate(body->callee());
                              //  Array<const Def*> new_args(body->num_args());
                              //  for ( size_t i = 0; i < body->num_args(); ++i)
                              //      new_args[i] = rewriter.instantiate(body->arg(i));
                              //  new_cont_->jump(new_callee, new_args, nkernel_cont->debug());


                                //nkernel_cont->set_interface();
                                //nkernel_cont->body()->callee()->as_nom<Continuation>()->attributes().interface = callee->interface();
                                //auto test = nkernel_cont->body()->callee()->as_nom<Continuation>()->interface();
                                auto test = nkernel_cont->get_interface();
                                if (test == Interface::Stream)
                                    std::cout << "NEW ASS STREAM" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }




    }

    );


 //   if (auto ncontinuation = importer.def_old2new_[ocontinuation]) {
 //       std::cout << "~~~~~~~~~~~~~~~~~" << std::endl;
 //       ocontinuation->dump();
 //       for ( auto use : ocontinuation->uses()) {
 //           if (auto app = use->isa<App>()) {
 //               auto test = app->using_continuations();
 //               for ( auto uc : test) {
 //                   std::cout << "######UC####" << std::endl;
 //                   uc->dump();
 //               }
 //           }
 //       }
 //   }

           //     for (auto use : global->uses()) {
           //         if (auto app = use->isa<App>()) {
           //             auto ucontinuations = app->using_continuations();
           //             for (const auto& block : target_blocks_in_hls_world) {
           //                 if (std::find(ucontinuations.begin(), ucontinuations.end(), block) != ucontinuations.end())
           //                     return true;
           //                 }
           //             }
           //         }





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


    std::vector<const Type*> graph_param_types;
    graph_param_types.emplace_back(world.mem_type());
    graph_param_types.emplace_back(world.fn_type({ world.mem_type() }));
    std::vector<std::tuple<Continuation*, size_t, size_t>> param_index; // tuples made of (new_kernel, index new kernel param, index graph param.)

    for (auto kernel : new_kernels) {
        for (size_t i = 0; i < kernel->num_params(); ++i) {
            auto param = kernel->param(i);
            // If the parameter is not a channel or is a channel but connected to a HLS kernel then
            // save the index and add it to the hls_top parameter list
            if (!is_channel_type(param->type())) {
                if (param != kernel->ret_param() && param != kernel->mem_param()) {
                    param_index.emplace_back(kernel, i, graph_param_types.size());
              //    top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    graph_param_types.emplace_back(param->type());
                }
            } else if (is_used_for_hls(param)) {
                    // finding the ports (channels connected to HLS kernels)
                    param_index.emplace_back(kernel, i, graph_param_types.size());
                    //top2kernel.emplace_back(top_param_types.size(), kernel->name(), i);
                    graph_param_types.emplace_back(param->type());
                    std::cout << "A PORT found! (on a kernel)" << std::endl;
                }
            }
        }

    //TODO: for the sake of simplicity and consistency with HLS at the moment we build IR similar to HLS top
    // but the proper way is to make a new continuaion for each kernel sependencies and jump to that accordingly.

    auto cgra_graph = world.continuation(world.fn_type(graph_param_types), Debug());
    cgra_graph->set_name("cgra_graph");
    //cgra_graph->attributes().interface = Interface::None;

    //auto struct_type = world.struct_type("hey",1);
    //struct_type->dump();
    // need variant type?
    //auto struct_ =  world.struct_agg(struct_type, {world.one(world.type_ps32())}, Debug("struct"));
    //auto struct_ =  world.struct_agg(struct_type, {world.literal_bool(true, Debug("test"))}, Debug("struct"));

    Def2Def global2param;
    for (auto tuple : param_index) {
        // Mapping cgra_graph params as args for new_kernels' params
        auto param = std::get<0>(tuple)->param(std::get<1>(tuple));
        auto arg   = cgra_graph->param(std::get<2>(tuple));
        if (is_used_for_hls(param)) {
            auto common_global = param2arg[param];
            global2param.emplace(common_global, param);
            // updating the args of the already inserted channel-params. Replacing these particular globals with channel-params.
            // note that emplace method only adds a new (keys,value) and does not update/rewrite values for already inserted keys
            param2arg[param] = arg;
            continue;
        }
        param2arg.emplace(param, arg); // adding (non-channel params, cgra_graph params as args). Channel params were added before
        //arg2param.emplace(arg, param); // channel-params are not here.
    }

    // send indices to codegen
    // these indices follow the hls-cgra dependent blocks order
    // and include only the indices for cgra kernel params that are connected to HLS kernels, basically ports
    auto hls_port_indices = external_ports_index(global2param, param2arg, def2dependent_blocks, importer);

    auto enter   = world.enter(cgra_graph->mem_param());
    auto cur_mem = world.extract(enter, 0_s);
    auto frame   = world.extract(enter, 1_s);

    // building slots only for channels(globals) between kernels
    Def2Def global2slot;
    for (const Def* prev_global; const auto& [_, arg] : param2arg) {
        if (arg->isa<Global>() && is_channel_type(arg->type()) && (arg != prev_global)) {
            prev_global = arg;
            const Def* channel_slot;
            channel_slot = world.slot(arg->type()->as<PtrType>()->pointee(), frame);
            global2slot.emplace(arg, channel_slot);
        }
    }
  //  //world.tuple_type({world.type_qs32(), world.type_qs32()});
  //  auto cont_type = world.fn_type({ world.mem_type() });
  //  auto d_cont_type = world.fn_type({ world.mem_type(), cont_type });
  //  auto dummy = world.continuation(d_cont_type, Debug());
  //  // TODO: use mem param of dummy to make slots then drop dummy and jump to kernels
  //  auto value = {world.one(world.type_qs32()), world.one(world.type_qs32())};
  //  auto tuple_value = world.tuple(value);
  //  auto slot = world.slot(world.tuple_type({world.type_qs32(), world.type_qs32()}), frame);
  //  auto slot_2 = world.slot(world.tuple_type({world.type_qs32(), world.type_qs32()}), frame);
  //  cur_mem = world.store(cgra_graph->mem_param(), slot, tuple_value, Debug());
  //  //TODO instead of calling kernels we can use slots with store but they are opt away...
  //  auto loaded = world.load(cur_mem, slot_2, Debug());
  //  auto elem_frame = world.extract(loaded, 1_u32, Debug());
  //  cur_mem = world.extract(loaded, 0_u32, Debug());

    //world.make_external(new_kernels.back());

    auto cur_bb = cgra_graph;
    for (const auto& kernel : new_kernels) {
        auto ret_param = kernel->ret_param();
        auto mem_param = kernel->mem_param();
        auto ret_type = ret_param->type()->as<FnType>();
        bool last_kernel = kernel == new_kernels.back();
        const Def* cgra_graph_ret = cgra_graph->ret_param();
        const Def* continuation = last_kernel ? cgra_graph_ret : world.continuation(ret_type, Debug("next_node"));

        Array<const Def*> args(kernel->type()->num_ops());
        for (size_t i = 0; i < kernel->type()->num_ops(); ++i) {
            auto param = kernel->param(i);
            if (param == mem_param) {
                args[i] = cur_mem;
            } else if (param == ret_param) {
                args[i] = continuation;
            } else if (auto arg = param2arg[param]) {
                args[i] = arg->isa<Global>() && is_channel_type(arg->type()) ? global2slot[arg] : arg;
            } else {
                assert(false);
            }
        }

        cur_bb->jump(kernel, args);

        if (!last_kernel) {
            auto next = continuation->as_nom<Continuation>();
            cur_bb = next;
            cur_mem = next->mem_param();
        }

        }

    world.make_external(cgra_graph);

    std::cout << "_--------cgra world After rewrite--------" <<std::endl;
    world.dump();
    for (auto def : world.defs()) {
        if (auto cont = def->isa_nom<Continuation>()) {
                auto gg = cont->get_interface();
                if (gg == Interface::Stream) {
                    std::cout << "MMMMMMMM" << std::endl;
                    cont->dump();
                }
        }
    }

    world.cleanup();

   // for (auto def : world.defs()) {
   //     if (auto cont = def->isa_nom<Continuation>()) {
   //         cont->set_interface();
   //     }
   // }
    return std::make_tuple(hls_port_indices, kernel_name2chan_param_modes);
}

}
