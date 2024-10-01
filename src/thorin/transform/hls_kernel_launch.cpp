#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/mangle.h"
#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/codegen.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static Continuation* make_opencl_intrinsic(World& world, const Continuation* cont_hls, const Array<const Def*> top_concrete_params) {
    assert(cont_hls->has_body());
    auto body = cont_hls->body();

    auto last_callee_continuation = body->arg(hls_free_vars_offset - 1)->isa_nom<Continuation>();
    auto kernel_ptr = body->arg(hls_free_vars_offset - 2);

    // building OpenCL intrinsics corresponding to hls intrinsic
    std::vector<const Type*> opencl_param_types;
    std::vector<const Type*> tuple_elems_type(3);

    //OpenCL--> fn(mem, device, grid, block, body, re_cont, /..../ )
    opencl_param_types.emplace_back(world.mem_type()); // mem
    opencl_param_types.emplace_back(world.type_qs32()); // device

    // grid and block
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0 ; j < tuple_elems_type.size() ; ++j)
            tuple_elems_type[j] = world.type_qs32();

        opencl_param_types.emplace_back(world.tuple_type(tuple_elems_type));
    }

    // type for dummy hls_top (body)
    opencl_param_types.emplace_back(kernel_ptr->type());
    // re_cont
    opencl_param_types.emplace_back(last_callee_continuation->type());

    // all parameters from device IR and the remaining
    std::transform(top_concrete_params.begin(), top_concrete_params.end(), std::back_inserter(opencl_param_types),
            [&](const Def* param) {

                if (is_channel_type(param->type())) {
                    auto struct_type = world.struct_type("channel", 1);
                    struct_type->set(0, world.type_bool());
                    return struct_type->as<Type>();
                }

                return param->type();
            });


    auto opencl_type = world.fn_type(opencl_param_types);

    auto opencl = world.continuation(opencl_type, Intrinsic::OpenCL, Debug("opencl"));
    return opencl;
}

static Continuation* last_basic_block_with_intrinsic(const Intrinsic intrinsic, const Schedule& schedule) {
    for (int i = schedule.size() - 1; i >= 0; --i) {
        auto block = schedule[i];
        if (!block->has_body()) continue;
        auto body = block->body();
        auto callee = body->callee()->isa_nom<Continuation>();
        if (callee && callee->intrinsic() == intrinsic) {
            return block;
        }
    }
    return nullptr;
}

const Def* has_hls_callee(Continuation* continuation) {
    assert(continuation->has_body());
    auto body = continuation->body();
    auto callee = body->callee()->isa_nom<Continuation>();
    if (callee && callee->intrinsic() == Intrinsic::HLS) {
        auto hls_cont_arg = body->arg(hls_free_vars_offset - 1);
        return hls_cont_arg;
    }
    return nullptr;
}

// Finds instances of HLS kernel launches and wraps them in a OpenCL launch
void hls_kernel_launch(World& world, HlsDeviceParams& device_params, Cont2Config& cont2config) {
    world.dump();

    auto hls_top_it = std::find_if(cont2config.begin(), cont2config.end(), [] (const auto& pair) {
        return pair.first->is_hls_top();
    });

    auto hls_top = (hls_top_it != cont2config.end()) ? hls_top_it->first : nullptr;
    assert(hls_top && "No hls_top found");


    // An array of hls_top parameters in which non-channel types are replaced with corresponding parameters in device_params
    // we assume that the orders of the parameters are the same in both hls_top and device_params
    // This is used to determine the location of OpenCL arguments
    const auto params_offset = 2; // mem and device are not considered as concrete parameters
    Array<const Def*> top_concrete_params(hls_top->num_params() - params_offset, nullptr);
    assert((hls_top->num_params() - params_offset) >= device_params.size() && " size of device non-channel params is greater than hls_top params");
    for (size_t i = params_offset, j = 0; i < hls_top->num_params(); ++i) {
        auto param = hls_top->param(i);
        top_concrete_params[i - params_offset] = is_channel_type(param->type()) ? param : device_params[j++];
    }


    bool last_hls_found = false;
    Continuation* opencl = nullptr;

    const size_t base_opencl_param_num = LaunchArgs<FPGA_CL>::Num;
    Array<const Def*> opencl_args(base_opencl_param_num + top_concrete_params.size());

    // TODO: perf opt, we only need to access the main scope
    // Maybe using world.externals() would be better
    Scope::for_each(world, [&] (Scope& scope) {

        Schedule scheduled = schedule(scope);
        for (auto& block : scheduled) {

            if (!block->has_body())
                continue;

            auto block_body = block->body();
            if (auto hls_callee = has_hls_callee(block)) {
                auto cont_mem_obj = block->mem_param();
                auto callee_continuation = hls_callee->isa_nom<Continuation>();
                Continuation* last_hls_cont;
                if (!last_hls_found) {
                    // TODO I'm at a loss for what is intended here. This is an assignment - not a check, the net result
                    // is the _only the first_ block with an HLS callee will enter this, which means the first block in the schedule
                    // will be rewired to call the opencl intrinsic, not the last. But it is still using the 'last' hls basic block
                    // as the one to pass to `make_opencl_intrinsic`. How is this meant to work ?
                    if ((last_hls_cont = last_basic_block_with_intrinsic(Intrinsic::HLS, scheduled)))
                        last_hls_found = true;

                    opencl = make_opencl_intrinsic(world, last_hls_cont, top_concrete_params);

                    // Building a dummy hls_top function
                    auto hls_top_fn_type = opencl->param(4)->type()->as<PtrType>()->pointee()->as<FnType>();
                    auto hls_top_fn = world.continuation(hls_top_fn_type, Debug("hls_top"));

                    auto hls_top_global = world.global(hls_top_fn, false);
                    opencl_args[4] = hls_top_global;

                    auto opencl_mem_param = opencl->mem_param();
                    auto opencl_device_param = opencl->param(1);

                    // Preparing basic argument for OpenCL call
                    // all other args are assigned from removed hls blocks
                    Array<const Def*> opencl_tuples_elems(opencl->param(2)->type()->num_ops());
                    for(size_t i = 0; i < opencl->num_params(); ++i) {
                        auto param = opencl->param(i);
                        if (param == opencl_mem_param)
                            opencl_args[i] = cont_mem_obj;
                        else if (param == opencl_device_param)
                            opencl_args[i] = block_body->arg(hls_free_vars_offset - 3);
                        else if (param->type()->isa<TupleType>()) {
                            // Block and grid fixed on 'one'
                            for (size_t j = 0; j < opencl_tuples_elems.size(); ++j) {
                                opencl_tuples_elems[j] = world.one(world.type_qs32());
                            }
                            opencl_args[i] = world.tuple(opencl_tuples_elems);
                        } else if (param->index() == 4 ) {
                            if (param->type() == opencl_args[4]->type())
                                // pointer to function is assigned where hls_top is created
                                continue;
                        } else if (param->type()->isa<FnType>() && param->order() == 1) {
                            opencl_args[i] = last_hls_cont->body()->arg(hls_free_vars_offset - 1);
                        } else if (!(param->type()->isa<StructType>()) && (i > base_opencl_param_num)) {
                            continue; // they are assigned in the next loop
                        } else if ((param->type()->isa<StructType>()) && (i > base_opencl_param_num)) {
                            auto bool_literal = {world.literal_bool(true, {"dummy_field"})};
                            auto dummy_struct =  world.struct_agg(param->type()->as<StructType>(), bool_literal, {"dummy_cgra_channel"});
                            opencl_args[i] = dummy_struct;
                        }
                    }
                }

                // extracting hls kernels' arguments
                // preparing OpenCL args
                for (size_t index = hls_free_vars_offset; index < block_body->num_args(); ++index) {
                    auto kernel = block_body->arg(2)->as<Global>()->init()->isa_nom<Continuation>();
                    auto kernel_param =  kernel->param(index - hls_free_vars_offset + 2);
                    // determining the correct location of OpenCL arguments by comparing kernels params with
                    // the location of hls_top params on device code (device_params)
                    auto param_on_device_it = std::find(top_concrete_params.begin(), top_concrete_params.end(), kernel_param);
                    size_t opencl_arg_index = std::distance(top_concrete_params.begin(), param_on_device_it);
                    assert(opencl_arg_index < top_concrete_params.size());
                    opencl_args[base_opencl_param_num + opencl_arg_index] = block_body->arg(index);
                }

                Array<const Def*> args(callee_continuation->type()->num_ops());
                for (size_t i = 0; i < callee_continuation->num_params(); ++i) {
                    auto param = block->param(i);
                    if (param == cont_mem_obj)
                        args[i] = cont_mem_obj;
                    else
                        args[i] = param;
                }
                // jump over all hls basic blocks until the last one
                // Replace the last one with the specialized basic block with Opencl intrinsic
                if (block != last_hls_cont) {
                    block->jump(callee_continuation, args);
                } else
                    block->jump(opencl, opencl_args);
            }

        }


    });

    debug_verify(world);
    world.dump();
    //world.cleanup();
}

}
