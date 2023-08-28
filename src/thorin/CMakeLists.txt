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
    analyses/domtree.cpp
    analyses/domtree.h
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
    transform/closure_conversion.cpp
    transform/closure_conversion.h
    transform/codegen_prepare.h
    transform/codegen_prepare.cpp
    transform/hoist_enters.cpp
    transform/hoist_enters.h
    transform/flatten_tuples.cpp
    transform/flatten_tuples.h
    transform/importer.cpp
    transform/importer.h
    transform/inliner.cpp
    transform/inliner.h
    transform/insert_pipeline_continue.cpp
    transform/insert_pipeline_continue.h
    transform/mangle.cpp
    transform/mangle.h
    transform/resolve_loads.cpp
    transform/resolve_loads.h
    transform/partial_evaluation.cpp
    transform/partial_evaluation.h
    transform/rewrite.cpp
    transform/rewrite.h
    transform/split_slots.cpp
    transform/split_slots.h
    transform/hls_channels.cpp
    transform/hls_channels.h
    transform/hls_kernel_launch.h
    transform/hls_kernel_launch.cpp
    transform/lower_control.h
    transform/lower_control.cpp
    transform/lower_return.h
    transform/lower_return.cpp
    transform/classify_control.h
    transform/classify_control.cpp
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
    util/graphviz_dump.cpp
    util/scoped_dump.h
    util/scoped_dump.cpp
    )

if(LLVM_FOUND)
    list(APPEND THORIN_SOURCES
        be/llvm/cpu.cpp
        be/llvm/cpu.h
        be/llvm/llvm.cpp
        be/llvm/llvm.h
        be/llvm/amdgpu.cpp
        be/llvm/amdgpu.h
        be/llvm/nvvm.cpp
        be/llvm/nvvm.h
        be/llvm/parallel.cpp
        be/llvm/runtime.inc
        be/llvm/runtime.cpp
        be/llvm/runtime.h
        be/llvm/vectorize.cpp
        be/llvm/debug.cpp
    )
endif()

if (shady_FOUND)
    list(APPEND THORIN_SOURCES
        be/shady/shady.cpp
    )
endif()

if(nlohmann_json_FOUND)
    list(APPEND THORIN_SOURCES
        be/json/json.cpp
        be/json/json.h
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

if (shady_FOUND)
    target_link_libraries(thorin PRIVATE shady::shady)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(thorin PRIVATE nlohmann_json::nlohmann_json)
endif()
