set(THORIN_SOURCES
    axiom.cpp
    axiom.h
    check.cpp
    check.h
    debug.cpp
    debug.h
    def.cpp
    def.h
    error.cpp
    error.h
    lam.cpp
    lam.h
    lattice.cpp
    lattice.h
    normalize.cpp
    normalize.h
    rewrite.cpp
    rewrite.h
    stream.cpp
    tuple.cpp
    tuple.h
    tables.h
    world.cpp
    world.h
    analyses/cfg.cpp
    analyses/cfg.h
    analyses/deptree.cpp
    analyses/deptree.h
    analyses/domfrontier.cpp
    analyses/domfrontier.h
    analyses/domtree.cpp
    analyses/domtree.h
    analyses/looptree.cpp
    analyses/looptree.h
    analyses/schedule.cpp
    analyses/schedule.h
    analyses/scope.cpp
    analyses/scope.h
    fe/lexer.cpp
    fe/lexer.h
    fe/parser.cpp
    fe/parser.h
    fe/tok.cpp
    fe/tok.h
    be/c.cpp
    be/c.h
    pass/optimize.cpp
    pass/pass.cpp
    pass/fp/eta_exp.cpp
    pass/fp/eta_exp.h
    pass/fp/eta_red.cpp
    pass/fp/eta_red.h
    pass/fp/beta_red.cpp
    pass/fp/beta_red.h
    pass/fp/copy_prop.cpp
    pass/fp/copy_prop.h
    pass/fp/dce.cpp
    pass/fp/dce.h
    pass/fp/ssa_constr.cpp
    pass/fp/ssa_constr.h
    pass/rw/auto_diff.cpp
    pass/rw/auto_diff.h
    pass/rw/partial_eval.cpp
    pass/rw/partial_eval.h
    pass/rw/ret_wrap.cpp
    pass/rw/ret_wrap.h
    pass/rw/bound_elim.cpp
    pass/rw/bound_elim.h
    pass/rw/scalarize.cpp
    pass/rw/scalarize.h
    transform/cleanup_world.cpp
    transform/cleanup_world.h
    transform/mangle.cpp
    transform/mangle.h
    transform/partial_evaluation.cpp
    transform/partial_evaluation.h
    transform/closure_conv.h
    transform/closure_conv.cpp
    transform/untype_closures.h
    transform/untype_closures.cpp
    util/array.h
    util/assert.h
    util/bit.h
    util/bitset.cpp
    util/bitset.h
    util/cast.h
    util/container.h
    util/hash.cpp
    util/hash.h
    util/indexmap.h
    util/indexset.h
    util/iterator.h
    util/ptr.h
    util/stream.cpp
    util/stream.h
    util/types.h
    util/utf8.cpp
    util/utf8.h
)

if(LLVM_FOUND)
    list(APPEND THORIN_SOURCES
        be/llvm/cpu.cpp
        be/llvm/cpu.h
        be/llvm/cuda.cpp
        be/llvm/cuda.h
        be/llvm/hls.cpp
        be/llvm/hls.h
        be/llvm/llvm.cpp
        be/llvm/llvm.h
        be/llvm/amdgpu.cpp
        be/llvm/amdgpu.h
        be/llvm/nvvm.cpp
        be/llvm/nvvm.h
        be/llvm/opencl.cpp
        be/llvm/opencl.h
        be/llvm/parallel.cpp
        be/llvm/runtime.inc
        be/llvm/runtime.cpp
        be/llvm/runtime.h
        be/llvm/vectorize.cpp
    )
endif()

add_library(libthorin ${THORIN_SOURCES})
set_target_properties(libthorin PROPERTIES PREFIX "")
target_compile_options(libthorin PRIVATE -Wall -Wextra)
target_include_directories(libthorin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(LLVM_FOUND)
    target_compile_definitions(libthorin PUBLIC ${LLVM_DEFINITIONS} LLVM_SUPPORT)
    target_include_directories(libthorin PRIVATE ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(LLVM_LIBRARIES all)
    target_link_libraries(libthorin PUBLIC ${LLVM_LIBRARIES})
endif()

if(RV_FOUND)
    target_link_libraries(libthorin PUBLIC ${RV_LIBRARIES})
    target_include_directories(libthorin PRIVATE ${RV_INCLUDE_DIRS})
endif()

find_package(Half REQUIRED)
message(STATUS "Building with Half library from ${Half_INCLUDE_DIRS}.")
target_include_directories(libthorin PUBLIC ${Half_INCLUDE_DIRS})

# check for colored terminal
set(COLOR_TTY_AVAILABLE TRUE)
if(WIN32)
    # Windows console does not support ANSI escape codes
    set(COLOR_TTY_AVAILABLE FALSE)
endif()
set(COLORIZE_OUTPUT ${COLOR_TTY_AVAILABLE} CACHE BOOL "Set to true to enable colored error messages")
if(COLORIZE_OUTPUT)
    target_compile_definitions(libthorin PUBLIC COLORIZE_LOG)
endif()
