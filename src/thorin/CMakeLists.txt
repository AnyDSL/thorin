set(THORIN_SOURCES
    continuation.cpp
    continuation.h
    debug.cpp
    debug.h
    def.cpp
    def.h
    enums.cpp
    enums.h
    primop.cpp
    primop.h
    rec_stream.cpp
    type.cpp
    type.h
    world.cpp
    world.h
    analyses/cfg.cpp
    analyses/cfg.h
    analyses/domfrontier.cpp
    analyses/domfrontier.h
    analyses/domtree.cpp
    analyses/domtree.h
    analyses/free_defs.cpp
    analyses/free_defs.h
    analyses/looptree.cpp
    analyses/looptree.h
    analyses/schedule.cpp
    analyses/schedule.h
    analyses/scope.cpp
    analyses/scope.h
    analyses/verify.cpp
    analyses/verify.h
    be/codegen.cpp
    be/codegen.h
    be/emitter.h
    be/c/c.cpp
    be/c/c.h
    be/kernel_config.h
    tables/allnodes.h
    tables/arithoptable.h
    tables/cmptable.h
    tables/nodetable.h
    tables/primtypetable.h
    tables/mathoptable.h
    transform/cleanup_world.cpp
    transform/cleanup_world.h
    transform/clone_bodies.cpp
    transform/clone_bodies.h
    transform/closure_conversion.cpp
    transform/closure_conversion.h
    transform/codegen_prepare.h
    transform/codegen_prepare.cpp
    transform/dead_load_opt.cpp
    transform/dead_load_opt.h
    transform/hoist_enters.cpp
    transform/hoist_enters.h
    transform/flatten_tuples.cpp
    transform/flatten_tuples.h
    transform/importer.cpp
    transform/importer.h
    transform/inliner.cpp
    transform/inliner.h
    transform/lift_builtins.cpp
    transform/lift_builtins.h
    transform/mangle.cpp
    transform/mangle.h
    transform/resolve_loads.cpp
    transform/resolve_loads.h
    transform/partial_evaluation.cpp
    transform/partial_evaluation.h
    transform/split_slots.cpp
    transform/split_slots.h
    transform/hls_channels.cpp
    transform/hls_channels.h
    transform/hls_kernel_launch.h
    transform/hls_kernel_launch.cpp
    util/array.h
    util/cast.h
    util/hash.h
    util/hash.cpp
    util/indexmap.h
    util/indexset.h
    util/iterator.h
    util/stream.cpp
    util/stream.h
    util/symbol.cpp
    util/symbol.h
    util/types.h
    util/utility.h
    )

if(LLVM_FOUND)
    list(APPEND THORIN_SOURCES
        be/llvm/cpu.cpp
        be/llvm/cpu.h
        be/llvm/llvm.cpp
        be/llvm/llvm.h
        be/llvm/amdgpu_hsa.cpp
        be/llvm/amdgpu_hsa.h
        be/llvm/amdgpu_pal.cpp
        be/llvm/amdgpu_pal.h
        be/llvm/nvvm.cpp
        be/llvm/nvvm.h
        be/llvm/parallel.cpp
        be/llvm/runtime.inc
        be/llvm/runtime.cpp
        be/llvm/runtime.h
        be/llvm/vectorize.cpp
    )
endif()

add_library(thorin ${THORIN_SOURCES})
target_include_directories(thorin PUBLIC ${Half_INCLUDE_DIRS} ${Thorin_ROOT_DIR}/src ${CMAKE_BINARY_DIR}/include)

if(LLVM_FOUND)
    set(Thorin_LLVM_COMPONENTS core support ipo target ${LLVM_TARGETS_TO_BUILD})
    target_include_directories(thorin SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(thorin PRIVATE ${LLVM_DEFINITIONS})
    if(RV_FOUND)
        target_include_directories(thorin PRIVATE ${RV_INCLUDE_DIRS})
        target_link_libraries(thorin PRIVATE ${RV_LIBRARIES})
        list(APPEND Thorin_LLVM_COMPONENTS analysis passes transformutils)
    endif()
    llvm_config(thorin ${AnyDSL_LLVM_LINK_SHARED} ${Thorin_LLVM_COMPONENTS})
endif()
