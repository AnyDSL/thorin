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

using Def2Mode = DefMap<ChannelMode>;

static void extract_kernel_channels(const Schedule& schedule, Def2Mode& def2mode) {
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
                        assert((!def2mode.contains(def) || def2mode[def] == ChannelMode::Write) &&
                                "Duplicated channel or \"READ\" mode channel redefined as WRITE!");
                        def2mode.emplace(def, ChannelMode::Write);
                    } else if (callee->name().str().find("read_channel") != std::string::npos) {
                        assert((!def2mode.contains(def) || def2mode[def] == ChannelMode::Read)  &&
                                "Duplicated channel or \"WRITE\" mode channel redefined as READ!");
                        def2mode.emplace(def, ChannelMode::Read);
                    } else {
                        ELOG("Not a channel / unsupported channel placeholder");
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

void hls_annotate_top(World& world, const Top2Kernel& top2kernel, Cont2Config& cont2config) {
    auto find_kernel_by_name = [&] (const std::string& name) {
        auto it = std::find_if(world.externals().begin(), world.externals().end(), [&] (auto external) {
            return external->name() == name;
        });
        return it != world.externals().end() ? *it : nullptr;
    };
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

void hls_channels(World& world, Top2Kernel& top2kernel) {
    std::vector<Def2Mode> channels_map; // vector of channel->mode maps for each kernel
    std::vector<Continuation*> new_kernels;
    Def2Def param2arg; // contains map from new kernel parameter to arguments of call inside hls_top (for all kernels)

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
        new_kernel->make_external();
        new_kernels.emplace_back(new_kernel);
        //kernel2index.emplace_back(new_kernel->name().str(), 0);
        old_kernel->make_internal();

        Rewriter rewriter;
        // Map the parameters of the old kernel to the first N parameters of the new one
        // The channels used inside the kernel are mapped to the parameters N + 1, N + 2, ...
        for (auto pair : index2def) {
            auto param = new_kernel->param(pair.first);
            rewriter.old2new[pair.second] = param;
            param2arg[param] = pair.second; // (channel params, globals)
        }
        for (auto def : scope.defs()) {
            if (auto cont = def->isa_continuation()) {
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
            if (auto cont = def->isa_continuation()) { // all basic blocks of the scope
                auto new_cont = rewriter.old2new[cont]->as_continuation();
                auto new_callee = rewriter.instantiate(cont->callee());
                Array<const Def*> new_args(cont->num_args());
                for ( size_t i = 0; i < cont->num_args(); ++i)
                    new_args[i] = rewriter.instantiate(cont->arg(i));
                new_cont->jump(new_callee, new_args, cont->debug());
            }
        }

        channels_map.emplace_back(def2mode);
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
                    top2kernel.emplace_back(top_param_types.size(), kernel->name().str(), i);
                    top_param_types.emplace_back(param->type());
                }
            }
        }
    }

    auto hls_top = world.continuation(world.fn_type(top_param_types), Debug("hls_top"));
    for (auto tuple : param_index) {
        // (non-channel params, top params as kernel call args)
        param2arg.emplace(std::get<0>(tuple)->param(std::get<1>(tuple)), hls_top->param(std::get<2>(tuple)));
    }

    auto enter   = world.enter(hls_top->mem_param());
    auto cur_mem = world.extract(enter, 0_s);
    auto frame   = world.extract(enter, 1_s);

    Def2Def global2slot;
    std::vector<const Def*> channel_slots;
    std::vector<const Global*> globals;
    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>())
            globals.emplace_back(global);
    }
    // We need to iterate over globals twice because we cannot iterate over primops while creating new primops
    for (auto global : globals) {
        if (is_channel_type(global->type())) {
            channel_slots.emplace_back(world.slot(global->type()->as<PtrType>()->pointee(), frame));
            global2slot.emplace(global, channel_slots.back());
        }
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

    hls_top->make_external();

    world.cleanup();
    world.dump();
}

}
