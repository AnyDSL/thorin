#include "lower_offload_intrinsics.h"

#include "runtime.h"

namespace thorin {

struct RuntimeAPI {
    World& world_;
    DeviceBackends& backends_;
    ContinuationMap<std::string> unique_kernel_names_;

    const Def* anydsl_alloc;
    const Def* anydsl_alloc_unified;
    const Def* anydsl_release;
    const Def* anydsl_launch_kernel;
    const Def* anydsl_parallel_for;
    const Def* anydsl_fibers_spawn;
    const Def* anydsl_spawn_thread;
    const Def* anydsl_sync_thread;
    const Def* anydsl_create_graph;
    const Def* anydsl_create_task;
    const Def* anydsl_create_edge;
    const Def* anydsl_execute_graph;

    std::string register_kernel_for_offloading(const App* launch, Continuation* kernel) {
        auto found = unique_kernel_names_.find(kernel);
        if (found != unique_kernel_names_.end())
            return found->second;
        kernel->set_name(kernel->unique_name());
        unique_kernel_names_[kernel] = kernel->name();
        backends_.register_kernel_for_offloading(launch, kernel);

        kernel->world().make_external(kernel);
        kernel->destroy("codegen");
        return kernel->name();
    }

    RuntimeAPI(World& world, DeviceBackends& backends) : world_(world), backends_(backends) {
        auto get_api_fn = [&](Types dom, const Type* codom, std::string name) {
            auto mem_ty = world.mem_type();
            auto found = world.find_cont(name.c_str());
            if (found)
                return found;
            auto r = codom ? world.return_type({mem_ty, codom}) : world.return_type({mem_ty});
            Array<const Type*> p = concat<const Type*>(mem_ty, concat<const Type*>(dom, r));
            auto c = world.continuation(world.fn_type(p), name);
            c->attributes_.cc = CC::C;
            world.make_external(c);
            return c;
        };

        auto i32 = world.type_qs32();
        auto i64 = world.type_qs64();
        auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_qu8()));

        anydsl_alloc = get_api_fn({ i32, i64 }, ptr_ty, "anydsl_alloc");
        anydsl_alloc_unified = get_api_fn({ i32, i64 }, ptr_ty, "anydsl_alloc_unified");
        anydsl_release = get_api_fn({ i32 }, nullptr, "anydsl_release");
        anydsl_launch_kernel = get_api_fn({ i32, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, i32 }, nullptr, "anydsl_launch_kernel");
        anydsl_parallel_for = get_api_fn({ i32, i32, i32, ptr_ty, ptr_ty }, nullptr, "anydsl_parallel_for");
        anydsl_fibers_spawn = get_api_fn({ i32, i32, i32, ptr_ty, ptr_ty }, nullptr, "anydsl_fibers_spawn");
        anydsl_spawn_thread = get_api_fn({ ptr_ty, ptr_ty }, i32, "anydsl_spawn_thread");
        anydsl_sync_thread = get_api_fn({ i32 }, nullptr, "anydsl_sync_thread");
        anydsl_create_graph = get_api_fn({ i32 }, i32, "anydsl_create_graph");
        anydsl_create_task = get_api_fn({ i32, world.tuple_type({ ptr_ty, i64 }) }, i32, "anydsl_create_task");
        anydsl_create_edge = get_api_fn({ i32, i32}, nullptr, "anydsl_create_edge");
        anydsl_execute_graph = get_api_fn({ i32, i32}, nullptr, "anydsl_execute_graph");
    }
};

static bool contains_ptrtype(const Type* type) {
    switch (type->tag()) {
        case Node_PtrType:             return false;
        case Node_IndefiniteArrayType: return contains_ptrtype(type->as<ArrayType>()->elem_type());
        case Node_DefiniteArrayType:   return contains_ptrtype(type->as<DefiniteArrayType>()->elem_type());
        case Node_FnType:              return false;
        case Node_StructType: {
            bool good = true;
            auto struct_type = type->as<StructType>();
            for (auto& t : struct_type->types())
                good &= contains_ptrtype(t);
            return good;
        }
        case Node_TupleType: {
            bool good = true;
            auto tuple = type->as<TupleType>();
            for (auto& t : tuple->types())
                good &= contains_ptrtype(t);
            return good;
        }
        default: return true;
    }
}

void emit_host_code(RuntimeAPI& api, const App* launch, Platform platform, const std::string& ext, Continuation* continuation) {
    World& world = continuation->world();

    assert(continuation->has_body());
    auto body = continuation->body();
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = body->callee()->as_nom<Continuation>();
    assert_unused(target->is_intrinsic());
    assert(body->num_args() >= KernelLaunchArgs::Num && "required arguments are missing");

    // arguments
    const Def* mem = body->arg(KernelLaunchArgs::Mem);
    auto ret = body->arg(KernelLaunchArgs::Return);

    auto target_device_id = body->arg(KernelLaunchArgs::Device);
    auto target_platform = world.literal_qs32(static_cast<uint32_t>(platform), {});
    auto target_device = world.arithop_or(target_platform, world.arithop_shl(target_device_id, world.literal_qs32(4, {})));

    auto it_space = body->arg(KernelLaunchArgs::Space);
    auto it_config = body->arg(KernelLaunchArgs::Config);
    auto kernel = body->arg(KernelLaunchArgs::Body)->as_nom<Continuation>();

    //auto kernel_name = builder.CreateGlobalString(kernel->name() == "hls_top" ? kernel->name() : kernel->name());
    auto kernel_name = world.global_immutable_string(api.register_kernel_for_offloading(launch, kernel));
    auto file_name = world.global_immutable_string(world.name() + ext);
    const size_t num_kernel_args = body->num_args() - KernelLaunchArgs::Num;

    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_qu8()));

    auto alloc = [&](const Type* t, std::string name) {
        auto a = world.alloc(t, mem, { name });
        mem = world.extract(a, static_cast<uint32_t>(0));
        return world.extract(a, 1);
    };

    auto store = [&](const Def* val, const Def* ptr) {
        mem = world.store(mem, ptr, val);
    };

    // allocate argument pointers, sizes, and types
    const Def* args   = alloc(world.definite_array_type(ptr_ty,                 num_kernel_args), "args");
    const Def* sizes  = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "sizes");
    const Def* aligns = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "aligns");
    const Def* allocs = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "allocs");
    const Def* types  = alloc(world.definite_array_type(world.type_qu8(),  num_kernel_args), "types");

    // fill array of arguments
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto target_arg = body->arg(i + KernelLaunchArgs::Num);
        //const auto target_val = code_gen.emit(target_arg);
        auto target_val = target_arg;

        KernelArgType arg_type;
        const Def* void_ptr;
        if (target_arg->type()->isa<DefiniteArrayType>() ||
            target_arg->type()->isa<StructType>() ||
            target_arg->type()->isa<TupleType>()) {
            // definite array | struct | tuple
            auto alloca = alloc(target_arg->type(), target_arg->name());
            store(target_val, alloca);

            // check if argument type contains pointers
            if (!contains_ptrtype(target_arg->type()))
                world.wdef(target_arg, "argument '{}' of aggregate type '{}' contains pointer (not supported in OpenCL 1.2)", target_arg, target_arg->type());

            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Struct;
        } else if (target_arg->type()->isa<PtrType>()) {
            auto ptr = target_arg->type()->as<PtrType>();
            auto rtype = ptr->pointee();

            //if (!rtype->isa<ArrayType>())
            //    world.edef(target_arg, "currently only pointers to arrays supported as kernel argument; argument has different type: {}", ptr);

            auto alloca = alloc(ptr_ty, target_arg->name());
            auto target_ptr = world.bitcast(ptr_ty, target_val);
            store(target_ptr, alloca);
            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Ptr;
        } else {
            // normal variable
            auto alloca = alloc(target_arg->type(), target_arg->name());
            store(target_val, alloca);

            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Val;
        }

        auto arg_ptr   = world.lea(args,   world.literal_pu32(i, {}), {});
        auto size_ptr  = world.lea(sizes,  world.literal_pu32(i, {}), {});
        auto align_ptr = world.lea(aligns, world.literal_pu32(i, {}), {});
        auto alloc_ptr = world.lea(allocs, world.literal_pu32(i, {}), {});
        auto type_ptr  = world.lea(types,  world.literal_pu32(i, {}), {});

        auto size = world.size_of(target_arg->type());

        //if (auto struct_type = llvm::dyn_cast<llvm::StructType>(target_val->getType())) {
        //    // In the case of a structure, do not include the padding at the end in the size
        //    auto last_elem   = struct_type->getStructNumElements() - 1;
        //    auto last_offset = layout_.getStructLayout(struct_type)->getElementOffset(last_elem);
        //    size = last_offset + layout_.getTypeStoreSize(struct_type->getStructElementType(last_elem)).getFixedValue();
        //}

        store(void_ptr, arg_ptr);
        store(size, size_ptr);
        store(world.align_of(target_arg->type()), align_ptr);
        store(world.size_of(target_arg->type()), alloc_ptr);
        store(world.literal_qu8((uint8_t)arg_type, {}), type_ptr);
    }

    // allocate arrays for the grid and block size
    const Def* grid_array = world.definite_array(world.type_qs32(), {
        world.extract(it_space, 0_u32),
        world.extract(it_space, 1_u32),
        world.extract(it_space, 2_u32),
    });
    const Def* grid_size = alloc(world.definite_array_type(world.type_qs32(), 3), "grid_size_alloc");
    store(grid_array, grid_size);

    const Def* block_array =  world.definite_array(world.type_qs32(), {
        world.extract(it_config, 0_u32),
        world.extract(it_config, 1_u32),
        world.extract(it_config, 2_u32),
    });
    const Def* block_size = alloc(world.definite_array_type(world.type_qs32(), 3), "block_size_alloc");
    store(block_array, block_size);

    grid_size  = world.bitcast(ptr_ty, grid_size);
    block_size = world.bitcast(ptr_ty, block_size);
    args       = world.bitcast(ptr_ty, args);
    sizes      = world.bitcast(ptr_ty, sizes);
    aligns     = world.bitcast(ptr_ty, aligns);
    allocs     = world.bitcast(ptr_ty, allocs);
    types      = world.bitcast(ptr_ty, types);

    file_name = world.bitcast(ptr_ty, file_name);
    kernel_name = world.bitcast(ptr_ty, kernel_name);

    continuation->set_body(world.app(api.anydsl_launch_kernel, {mem, target_device, file_name, kernel_name, grid_size, block_size, args, sizes, aligns, allocs, types, world.literal_qs32(num_kernel_args, {}), ret}));
}

std::tuple<const Def*, Array<const Def*>> spill(const Def*& mem, const Defs& args, const Def*& wrapper_mem, const Def* wrapper_ptr) {
    World& world = mem->world();
    StructType* st = world.struct_type("spillbox", args.size());
    for (size_t i = 0; i < args.size(); i++)
        st->set_op(i, args[i]->type());

    auto alloc = [&](const Type* t, std::string name) {
        auto a = world.alloc(t, mem, { name });
        mem = world.extract(a, static_cast<uint32_t>(0));
        return world.extract(a, 1);
    };

    auto store = [&](const Def* val, const Def* ptr) {
        mem = world.store(mem, ptr, val);
    };

    const Def* spill_alloca = alloc(st, "spill");
    std::vector<const Def*> restored;
    wrapper_ptr = world.bitcast(spill_alloca->type(), wrapper_ptr);
    for (size_t i = 0; i < args.size(); i++) {
        store(args[i], world.lea(spill_alloca, world.literal_pu32(i, {}), {}));
        auto l = world.load(wrapper_mem, world.lea(wrapper_ptr, world.literal_pu32(i, {}), {}));
        wrapper_mem = world.extract(l, static_cast<uint32_t>(0));
        restored.push_back(world.extract(l, 1));
    }

    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_qu8()));
    return std::make_tuple(world.bitcast(ptr_ty, spill_alloca), restored);
}

enum class RuntimeParallelForArgs {
    Mem = 0,
    NumThreads,
    Lower,
    Upper,
    Args,
    Fun,
    Return,
};

void emit_parallel(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_qu8()));

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(ParallelForArgs::Mem));
    auto numthreads = body->arg(static_cast<size_t>(ParallelForArgs::NumThreads));
    auto lower = body->arg(static_cast<size_t>(ParallelForArgs::Lower));
    auto upper = body->arg(static_cast<size_t>(ParallelForArgs::Upper));
    auto fun = body->arg(static_cast<size_t>(ParallelForArgs::Fun))->as<Continuation>();
    auto ret = body->arg(static_cast<size_t>(ParallelForArgs::Return));

    // create loop iterating over range:
    // for (int i=lower; i<upper; ++i)
    //   body(i, <closure_elems>);

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, world.type_qs32(), world.type_qs32(), world.return_type({world.mem_type()})}));
    world.make_external(wrapper);
    const Def* wrapper_mem = wrapper->mem_param();
    auto inner_lower = wrapper->param(2);
    auto inner_upper = wrapper->param(3);
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(ParallelForArgs::Num)), wrapper_mem, wrapper->param(1));

    auto loop_head = world.continuation(world.fn_type({world.mem_type(), world.type_qs32()}), "loop_head");
    auto loop_body = world.continuation(world.fn_type({world.mem_type()}), "loop_body");
    auto loop_continue = world.continuation(world.fn_type({world.mem_type()}), "loop_continue");
    auto loop_exit = world.continuation(world.fn_type({world.mem_type()}), "loop_exit");
    loop_head->branch(loop_head->mem_param(), world.cmp_lt(loop_head->param(1), inner_upper), loop_body, loop_exit);

    Array<const Def*> prefix {loop_body->mem_param(), world.bitcast(fun->param(1)->type(), loop_head->param(1)), world.return_point(loop_continue) };
    loop_body->jump(fun, concat(prefix, recovered));
    loop_continue->jump(loop_head, { loop_continue->mem_param(), world.arithop_add(loop_head->param(1), world.literal_qs32(1, {})) });
    loop_exit->jump(wrapper->ret_param(), loop_exit->params_as_defs());
    wrapper->jump(loop_head, { wrapper->mem_param(), inner_lower });

    continuation->set_body(world.app(api.anydsl_parallel_for, {mem, numthreads, lower, upper, world.bitcast(ptr_ty, args), world.bitcast(ptr_ty, wrapper), ret}));
}

enum class RuntimeSpawnFibersArgs {
    Mem = 0,
    NumThreads,
    NumBlocks,
    NumWarps,
    Args,
    Fun,
    Return,
};

void emit_fibers(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.type_qu8());
    auto i32 = world.type_qs32();

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SpawnFibersArgs::Mem));
    auto threads = body->arg(static_cast<size_t>(SpawnFibersArgs::NumThreads));
    auto blocks = body->arg(static_cast<size_t>(SpawnFibersArgs::NumBlocks));
    auto warps = body->arg(static_cast<size_t>(SpawnFibersArgs::NumWarps));
    auto fun = body->arg(static_cast<size_t>(SpawnFibersArgs::Fun))->as<Continuation>();
    auto ret = body->arg(static_cast<size_t>(SpawnFibersArgs::Return));

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, i32, i32, world.return_type({world.mem_type()})}));
    const Def* wrapper_mem = wrapper->mem_param();
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(SpawnThreadArgs::Num)), wrapper_mem, wrapper->param(1));
    Array<const Def*> prefix = {mem, wrapper->param(2), wrapper->param(3)};
    wrapper->jump(fun, concat<const Def*>(concat(prefix, recovered), {wrapper->ret_param()}));

    continuation->set_body(world.app(api.anydsl_fibers_spawn, {mem, threads, blocks, warps, args, fun, ret}));
}

enum class RuntimeSpawnThreadArgs {
    Mem = 0,
    Args,
    Fun,
    Return,
};

void emit_spawn(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_qu8()));

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SpawnThreadArgs::Mem));
    auto fun = body->arg(static_cast<size_t>(SpawnThreadArgs::Fun));
    auto ret = body->arg(static_cast<size_t>(SpawnThreadArgs::Return));

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, world.return_type({world.mem_type()})}));
    const Def* wrapper_mem = wrapper->mem_param();
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(SpawnThreadArgs::Num)), wrapper_mem, wrapper->param(1));
    wrapper->jump(fun, concat<const Def*>(concat<const Def*>(mem, recovered.ref()), {wrapper->ret_param()}));

    continuation->set_body(world.app(api.anydsl_sync_thread, {mem, world.bitcast(ptr_ty, args), world.bitcast(ptr_ty, wrapper), ret}));
}

void emit_sync(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SyncArgs::Mem));
    auto id = body->arg(static_cast<size_t>(SyncArgs::Id));
    auto ret = body->arg(static_cast<size_t>(SyncArgs::Return));

    continuation->set_body(world.app(api.anydsl_sync_thread, {mem, id, ret}));
}

void lower_offload_intrinsics(World& world, DeviceBackends& backends) {
    RuntimeAPI api(world, backends);

    for (auto continuation : world.copy_continuations()) {
        if (!continuation->has_body()) continue;
        auto call = continuation->body();
        if (auto callee = call->callee()->isa<Continuation>()) {
            switch (callee->intrinsic()) {
                case Intrinsic::CUDA:            emit_host_code(api, call, Platform::CUDA_PLATFORM,       ".cu",     continuation); break;
                case Intrinsic::NVVM:            emit_host_code(api, call, Platform::CUDA_PLATFORM,       ".nvvm",   continuation); break;
                case Intrinsic::OpenCL:          emit_host_code(api, call, Platform::OPENCL_PLATFORM,     ".cl",     continuation); break;
                case Intrinsic::OpenCL_SPIRV:    emit_host_code(api, call, Platform::OPENCL_PLATFORM,     ".spv",    continuation); break;
                case Intrinsic::LevelZero_SPIRV: emit_host_code(api, call, Platform::LEVEL_ZERO_PLATFORM, ".spv",    continuation); break;
                case Intrinsic::AMDGPUHSA:       emit_host_code(api, call, Platform::HSA_PLATFORM,        ".amdgpu", continuation); break;
                case Intrinsic::AMDGPUPAL:       emit_host_code(api, call, Platform::PAL_PLATFORM,        ".amdgpu", continuation); break;
                case Intrinsic::VulkanCS_SPIRV:  emit_host_code(api, call, Platform::VULKAN_PLATFORM,     ".spv",    continuation); break;

                case Intrinsic::Parallel:        emit_parallel(api, continuation); break;
                case Intrinsic::Fibers:          emit_fibers(api, continuation);   break;
                case Intrinsic::Spawn:           emit_spawn(api, continuation);    break;
                case Intrinsic::Sync:            emit_sync(api, continuation);     break;
                default: continue;
            }
        }
    }
}

}
