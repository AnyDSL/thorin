#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/mangle.h"
#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

namespace thorin {

static void extract_opencl_args(Continuation* cont, Array<const Def*>& args) {
    // variables (arguments in this context) are placed after (mem, dev, lambda, continuation, /vars_args/ )
    const char vars_offset = 4;
    size_t var_num_args = cont->num_args() - vars_offset;
    if (var_num_args == 0)
        return;
    else {
        for (size_t index = vars_offset; index < cont->num_args(); ++index) {
            std::cout << index - vars_offset + var_num_args << endl;
            args[index - vars_offset + var_num_args] = cont->arg(index);
            }
        }
    }

static Continuation* make_opencl(World& world, const Continuation* cont_hls, const DeviceParams& device_params) {

    auto last_callee_continuation = cont_hls->arg(hls_free_vars_offset - 1)->as_continuation();
    auto last_callee_body = cont_hls->arg(hls_free_vars_offset - 2);

    // building OpenCL intrinsics corresponding to hls intrinsic
    std::vector<const Type*> opencl_param_types;
    std::vector<const Type*> tuple_elems_type(3);

    //OpenCL--> fn(mem, device, grid, block, body, re_cont, /..../ )
    opencl_param_types.emplace_back(world.mem_type());
    opencl_param_types.emplace_back(world.type_qs32());
    //  device and grid types
    for (size_t i = 0; i < 2 ; ++i) {
        for (size_t i = 0 ; i < tuple_elems_type.size() ; ++i)
            tuple_elems_type[i] = world.type_qs32();

        opencl_param_types.emplace_back(world.tuple_type(tuple_elems_type));
    }

    // type for dummy hls_top
    opencl_param_types.emplace_back(last_callee_body->type());

    opencl_param_types.emplace_back(last_callee_continuation->type());

    // all parameters from device IR and the remaining
    for (auto def : device_params)
        opencl_param_types.emplace_back(def->type());

    auto opencl_type = world.fn_type(opencl_param_types);

    auto opencl = world.continuation(opencl_type, CC::C, Intrinsic::OpenCL, Debug("opencl"));
    return opencl;
}

static Continuation* last_cl_kernel(const Schedule& schedule) {
    const auto& scheduled_blocks = schedule.blocks();
    for (size_t i = schedule.size() - 1; i >= 0; --i) {
        auto continuation = scheduled_blocks[i].continuation();
        auto callee = continuation->callee()->isa_continuation();
        if (callee && callee->intrinsic() == Intrinsic::OpenCL) {
            return continuation;
        }
    }
    return nullptr;
}

// Returns last Basic Block with corresponding intrinsic
static Continuation* last_basic_block(const Intrinsic intrinsic, const Schedule& schedule) {
    const auto& scheduled_blocks = schedule.blocks();
    for (size_t i = schedule.size() - 1; i >= 0; --i) {
        auto continuation = scheduled_blocks[i].continuation();
        auto callee = continuation->callee()->isa_continuation();
        if (callee && callee->intrinsic() == intrinsic) {
            return continuation;
        }
    }
    return nullptr;
}

Continuation* is_opencl(const Schedule schedule) {
    for (auto& block : schedule) {
        auto continuation = block.continuation();
        if (continuation->empty())
            continue;
        auto callee = continuation->callee()->isa_continuation();
        if (callee && callee->intrinsic() == Intrinsic::OpenCL) {
            return continuation;
        }
    }
    return nullptr;
}

Continuation* is_hls(const Schedule schedule) {
    for (auto& block : schedule) {
        auto continuation = block.continuation();
        if (continuation->empty())
            continue;
        auto callee = continuation->callee()->isa_continuation();
        if (callee && callee->intrinsic() == Intrinsic::HLS) {
            return continuation;
        }
    }
    return nullptr;
}

Continuation* is_opencl(Continuation* continuation) {
    auto callee = continuation->callee()->isa_continuation();
    if (callee && callee->intrinsic() == Intrinsic::OpenCL) {
        return continuation;
    }
    return nullptr;
}

// Returns a pointer to the first continuation calling OpenCL and the callee continuation
std::unique_ptr<std::pair<Continuation*, Continuation*>> has_opencl_callee(Continuation* continuation) {
    auto callee = continuation->callee()->isa_continuation();
    if (callee && callee->intrinsic() == Intrinsic::OpenCL) {
        auto opencl_cont = continuation->arg(5)->as_continuation();
        return std::make_unique<std::pair<Continuation*, Continuation*>> (continuation, opencl_cont);
    }
    return nullptr;
}

// Returns a pointer to the first continuation calling hls and the continuation of hls callee
std::unique_ptr<std::pair<Continuation*, const Def*>> has_hls_callee(Continuation* continuation) {
    auto callee = continuation->callee()->isa_continuation();
    if (callee && callee->intrinsic() == Intrinsic::HLS) {
        auto hls_cont_arg = continuation->arg(hls_free_vars_offset - 1);
        return std::make_unique<std::pair<Continuation*, const Def*>> (continuation, hls_cont_arg);
    }
    return nullptr;
}

void hls_kernel_launch(World& world, DeviceParams& device_params) {
    std::cout << "*** called ***"<< endl;

    Continuation* opencl = nullptr;
    const size_t base_opencl_param_num = 6;
    Array<const Def*> opencl_args(base_opencl_param_num + device_params.size());

    bool last_hls_found = false;
    Scope::for_each(world, [&] (Scope& scope) {
            Schedule schedule(scope);

            for (auto& block : schedule) {
                auto basic_block = block.continuation();

                if (basic_block->empty())
                    continue;

                if (auto conts_ptr = has_hls_callee(basic_block)) {
                    auto cont_mem_obj = conts_ptr->first->mem_param();
                    auto callee_continuation = conts_ptr->second->as_continuation();
                    Continuation* last_hls_cont;
                    if (!last_hls_found) {
                        if ( (last_hls_cont = last_basic_block(Intrinsic::HLS, schedule)) )
                            last_hls_found = true;

                        opencl = make_opencl(world, last_hls_cont, device_params);

                        // Building a dummy hls_top function
                        auto hls_top_fn_type = opencl->param(4)->type()->as<PtrType>()->pointee()->as<FnType>();
                        auto hls_top_fn = world.continuation(hls_top_fn_type, Debug("hls_top"));

                        auto hls_top_global = world.global(hls_top_fn,false);
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
                            else if (param == opencl_device_param) {
                                opencl_args[i] = conts_ptr->first->arg(hls_free_vars_offset - 3);

                            } else if (param->type()->isa<TupleType>()) {
                                // Block and grid fixed on 'one'
                                for (size_t j = 0; j < opencl_tuples_elems.size(); ++j) {
                                    opencl_tuples_elems[j] = world.one(world.type_qs32());
                                }
                                opencl_args[i] = world.tuple(opencl_tuples_elems);
                            } else if (param->index() == 4 ) {
                                if (param->type() == opencl_args[4]->type())
                                    // pointer to function is assigned where hls_top is created
                                    continue;
                            } else if ( param->type()->isa<FnType>() && param->order() == 1) {
                                opencl_args[i] = last_hls_cont->arg(hls_free_vars_offset - 1);
                            }

                        }
                        }

                    // extracting hls kernels' arguments
                    // preparing OpenCL args
                    for (size_t index = hls_free_vars_offset; index < basic_block->num_args(); ++index) {
                        //auto kernel = basic_block->arg(1)->as<Global>()->init()->as_continuation();
                        auto kernel = basic_block->arg(2)->as<Global>()->init()->as_continuation();
                        auto kernel_param =  kernel->param(index - hls_free_vars_offset + 2);
                        // determining the correct location of OpenCL arguments by comparing kernels params with
                        // the location of hls_top params on device code (device_params)
                        auto param_on_device_it = std::find(device_params.begin(), device_params.end(), kernel_param);
                        size_t opencl_arg_index = std::distance(device_params.begin(), param_on_device_it);
                        assert(opencl_arg_index < device_params.size());
                        opencl_args[opencl_arg_index + base_opencl_param_num] = basic_block->arg(index);

                    }

                auto cur_bb  = basic_block;
                auto cur_mem = cont_mem_obj;

                Array<const Def*> args(callee_continuation->type()->num_ops());
                for (size_t i = 0; i < callee_continuation->num_params(); ++i) {
                    auto param = basic_block->param(i);
                    if (param == cont_mem_obj)
                        args[i] = cur_mem;
                    else
                        args[i] = param;
                }
                // jump over all hls basic blocks until the last one
                // Replace the last one with the specialized basic block with Opencl intrinsic
                if (basic_block != last_hls_cont)
                    cur_bb->jump(callee_continuation, args);
                  else
                    cur_bb->jump(opencl,opencl_args);
            }
        }

    });

    debug_verify(world);
    std::cout << endl << "************ MODULE *************" << endl;
    world.dump();
    world.cleanup();
}

}
