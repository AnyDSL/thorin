#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

using Def2Mode = DefMap<ChannelMode>;
using Dependencies = std::vector<std::pair<size_t, size_t>>; // <From, To>

static void extract_kernel_channels(const Schedule& schedule, Def2Mode& def2mode) {
    for (const auto& continuation : schedule) {

        if (!continuation->has_body())
            continue;
        auto app = continuation->body();

        auto callee = app->callee()->isa_nom<Continuation>();
        if (callee && callee->is_channel()) {
            if (app->arg(1)->order() == 0 && !(is_mem(app->arg(1)) || is_unit(app->arg(1)))) {
                auto def= app->arg(1);
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
        auto param  = kernel->param(std::get<2>(tuple));
        assert(kernel);
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

/**
 * @param importer hls world
 * @param Top2Kernel annonating hls_top configuration
 * @param old_world to connect with runtime (host) world
 * @return corresponding hls_top parameter for hls_launch_kernel in another world (params before rewriting kernels)
 */

DeviceParams hls_channels(Importer& importer, Top2Kernel& top2kernel, World& old_world) {
    auto& world = importer.world();
    std::vector<Def2Mode> kernels_ch_modes; // vector of channel->mode maps for kernels which use channel(s)
    std::vector<Continuation*> new_kernels;
    Def2Def kernel_new2old;
    Def2Def param2arg; // contains map from new kernel parameter to arguments of calls inside hls_top (for all kernels)
    Def2Def arg2param;


    Scope::for_each(world, [&] (Scope& scope) {
            auto old_kernel = scope.entry();
            Def2Mode def2mode;
            extract_kernel_channels(schedule(scope), def2mode);

            Array<const Type*> new_param_types(def2mode.size() + old_kernel->num_params());
            std::copy(old_kernel->type()->ops().begin(),
                    old_kernel->type()->ops().end(),
                    new_param_types.begin());
            size_t i = old_kernel->num_params();
            // This vector records pairs containing:
            // - The position of the channel parameter for the new kernel
            // - The old global definition for the channel
            std::vector<std::pair<size_t, const Def*>> index2def;
            for (auto map : def2mode) {
                index2def.emplace_back(i, map.first);
                new_param_types[i++] = map.first->type();
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
                    assert(cont->has_body());
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
        // (non-channel params, top params as kernel call args)
        auto param = std::get<0>(tuple)->param(std::get<1>(tuple));
        auto arg   = hls_top->param(std::get<2>(tuple));
        param2arg.emplace(param, arg);
        arg2param.emplace(arg, param);
    }

    // ---------- Preparing args for calling hls_top from host ------------

    // Maping new_kernels params to old kernels params
    std::vector<const Def*> old_kernels_params;
    for (auto param : hls_top->params()) {
        if (arg2param.contains(param)) {
            auto new_kernel_param = arg2param[param]->as<Param>();
            auto old_kernel = kernel_new2old[new_kernel_param->continuation()];
            old_kernels_params.emplace_back(old_kernel->as_nom<Continuation>()->param(new_kernel_param->index()));
        }
    }

    // Maping hls world params (from old kernels) to old_world params. Required for host code (runtime) generation
    for (auto& elem : old_kernels_params) {
        for (auto def : old_world.defs()) {
            if (auto ocontinuation = def->isa_nom<Continuation>()) {
                auto ncontinuation = elem->as<Param>()->continuation();
                if (ncontinuation == importer.def_old2new_[ocontinuation]) {
                    elem = ocontinuation->param(elem->as<Param>()->index());
                    break;
                }
            }
        }
    }

    // --------------------------------------------------------------------

    auto enter   = world.enter(hls_top->mem_param());
    auto cur_mem = world.extract(enter, 0_s);
    // hls_top memory obj frame to be used in making channel slots
    auto frame   = world.extract(enter, 1_s);

    Def2Def global2slot;
    std::vector<const Def*> channel_slots;
    std::vector<const Global*> globals;
    for (auto def : world.defs()) {
        if (auto global = def->isa<Global>())
            globals.emplace_back(global);
    }

    Dependencies dependencies;
    // We need to iterate over globals twice because we cannot iterate over primops while creating new primops
    for (auto global : globals) {
        if (is_channel_type(global->type())) {
            channel_slots.emplace_back(world.slot(global->type()->as<PtrType>()->pointee(), frame));
            global2slot.emplace(global, channel_slots.back());
        }

        // Finding all dependencies between the kernels
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
        auto ret = world.continuation(ret_type, kernel->debug());
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
        cur_bb = ret;
        cur_mem = ret->mem_param();
    }

    world.make_external(hls_top);

    debug_verify(world);
    world.cleanup();

    return old_kernels_params;
}

}
