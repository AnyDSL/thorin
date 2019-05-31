#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/mangle.h"
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
Def2Mode  def2mode_;

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
                        assert((!def2mode_.contains(def) || def2mode_[def] == ChannelMode::Write) && "Duplicated channel or \"READ\" mode channel redefined as WRITE!");
                        def2mode_.emplace(def,ChannelMode::Write);
                    } else if (callee->name().str().find("read_channel") != std::string::npos) {
                        assert((!def2mode_.contains(def) || def2mode_[def] == ChannelMode::Read)  && "Duplicated channel or \"WRITE\" mode channel redefined as READ!");
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
    Scope::for_each(world, [&] (Scope& scope) {
        auto kernel = scope.entry();
        extract_kernel_channels(schedule(scope));

        Array<const Type*> new_param_types(def2mode_.size() + kernel->num_params());
        std::copy(kernel->type()->ops().begin(),
                  kernel->type()->ops().end(),
                  new_param_types.begin());
        size_t i = scope.entry()->num_params();
        std::vector<std::pair<size_t, const Def*>> index2def;
        for (auto map : def2mode_) {
            index2def.emplace_back(i, map.first); // (index, global) pairs for each kernel
            new_param_types[i++] = map.first->type();
        }
        auto new_kernel = world.continuation(world.fn_type(new_param_types), kernel->debug());
        new_kernel->make_external();
        kernel->make_internal();

        Rewriter rewriter;
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
       //each cont have a callee and a set of ops (here args), we should jump to new callees with new args
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
        def2mode_.clear();
    });

    world.dump();
    world.cleanup();
}







}
