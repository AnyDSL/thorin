set(THORIN_SOURCES
    continuation.cpp
    debug.cpp
    def.cpp
    enums.cpp
    primop.cpp
    rec_stream.cpp
    type.cpp
    world.cpp
    analyses/cfg.cpp
    analyses/domtree.cpp
    analyses/free_defs.cpp
    analyses/looptree.cpp
    analyses/schedule.cpp
    analyses/scope.cpp
    analyses/verify.cpp
    be/codegen.cpp
    be/c/c.cpp
    transform/cleanup_world.cpp
    transform/closure_conversion.cpp
    transform/codegen_prepare.cpp
    transform/dead_load_opt.cpp
    transform/hls_channels.cpp
    transform/hls_kernel_launch.cpp
    transform/hoist_enters.cpp
    transform/flatten_tuples.cpp
    transform/importer.cpp
    transform/inliner.cpp
    transform/lift_builtins.cpp
    transform/mangle.cpp
    transform/partial_evaluation.cpp
    transform/resolve_loads.cpp
    transform/rewrite.cpp
    transform/split_slots.cpp
    util/hash.cpp
    util/stream.cpp
    util/symbol.cpp
    util/graphviz_dump.cpp
    util/scoped_dump.cpp
    )

if(LLVM_FOUND)
    list(APPEND THORIN_SOURCES
        be/llvm/cpu.cpp
        be/llvm/llvm.cpp
        be/llvm/amdgpu.cpp
        be/llvm/amdgpu_hsa.cpp
        be/llvm/amdgpu_pal.cpp
        be/llvm/nvvm.cpp
        be/llvm/parallel.cpp
        be/llvm/runtime.inc
        be/llvm/runtime.cpp
        be/llvm/vectorize.cpp
    )
endif()

if (THORIN_ENABLE_SHADY)
    list(APPEND THORIN_SOURCES
        be/shady/shady.cpp
    )
endif()

if(THORIN_ENABLE_JSON)
    list(APPEND THORIN_SOURCES
        be/json/json.cpp
    )
endif()

add_library(thorin ${THORIN_SOURCES})
target_include_directories(thorin PUBLIC ${Half_INCLUDE_DIRS} ${Thorin_ROOT_DIR}/src ${Thorin_ROOT_DIR}/include ${CMAKE_BINARY_DIR}/include)

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

if (THORIN_ENABLE_SHADY)
    if (shady_FOUND)
        target_link_libraries(thorin PRIVATE shady::shady)
    else()
        target_link_libraries(thorin PRIVATE shady)
    endif()
endif()

if(THORIN_ENABLE_JSON)
    target_link_libraries(thorin PRIVATE nlohmann_json::nlohmann_json)
endif()
