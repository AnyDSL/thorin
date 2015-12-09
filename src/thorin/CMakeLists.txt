SET ( LIBRARY_NAME thorin )

# get source files
SET ( SOURCES
    def.cpp
    def.h
    enums.cpp
    enums.h
    irbuilder.cpp
    irbuilder.h
    lambda.cpp
    lambda.h
    primop.cpp
    primop.h
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
    analyses/free_vars.cpp
    analyses/free_vars.h
    analyses/liveness.cpp
    analyses/liveness.h
    analyses/looptree.cpp
    analyses/looptree.h
    analyses/schedule.cpp
    analyses/schedule.h
    analyses/scope.cpp
    analyses/scope.h
    analyses/verify.cpp
    analyses/verify.h
    be/c.cpp
    be/c.h
    be/llvm/cpu.cpp
    be/llvm/cpu.h
    be/llvm/cuda.cpp
    be/llvm/cuda.h
    be/llvm/llvm.cpp
    be/llvm/llvm.h
    be/llvm/nvvm.cpp
    be/llvm/nvvm.h
    be/llvm/opencl.cpp
    be/llvm/opencl.h
    be/llvm/parallel.cpp
    be/llvm/runtime.cpp
    be/llvm/runtime.h
    be/llvm/runtimes/cuda_runtime.cpp
    be/llvm/runtimes/cuda_runtime.h
    be/llvm/runtimes/generic_runtime.cpp
    be/llvm/runtimes/generic_runtime.h
    be/llvm/runtimes/nvvm_runtime.cpp
    be/llvm/runtimes/nvvm_runtime.h
    be/llvm/runtimes/opencl_runtime.cpp
    be/llvm/runtimes/opencl_runtime.h
    be/llvm/runtimes/spir_runtime.cpp
    be/llvm/runtimes/spir_runtime.h
    be/llvm/spir.cpp
    be/llvm/spir.h
    be/llvm/vectorize.cpp
    be/ycomp.cpp
    be/ycomp.h
    tables/allnodes.h
    tables/arithoptable.h
    tables/cmptable.h
    tables/nodetable.h
    tables/primtypetable.h
    transform/cleanup_world.cpp
    transform/cleanup_world.h
    transform/clone_bodies.cpp
    transform/clone_bodies.h
    transform/critical_edge_elimination.cpp
    transform/critical_edge_elimination.h
    transform/dead_load_opt.cpp
    transform/dead_load_opt.h
    transform/import.cpp
    transform/import.h
    transform/inliner.cpp
    transform/inliner.h
    transform/lift_builtins.cpp
    transform/lift_builtins.h
    transform/lift_enters.cpp
    transform/lift_enters.h
    transform/lower2cff.cpp
    transform/lower2cff.h
    transform/mangle.cpp
    transform/mangle.h
    transform/mem2reg.cpp
    transform/mem2reg.h
    transform/memmap_builtins.cpp
    transform/memmap_builtins.h
    transform/partial_evaluation.cpp
    transform/partial_evaluation.h
    util/args.h
    util/array.h
    util/assert.h
    util/autoptr.h
    util/cast.h
    util/hash.h
    util/indexmap.h
    util/indexset.h
    util/iterator.h
    util/location.cpp
    util/location.h
    util/log.cpp
    util/log.h
    util/printer.cpp
    util/printer.h
    util/push.h
    util/queue.h
    util/stdlib.h
    util/stream.cpp
    util/stream.h
    util/types.h
    util/ycomp.cpp
    util/ycomp.h
)

# create lib
get_thorin_dependency_libs( THORIN_LIBS )

# set define for *.s files
GET_FILENAME_COMPONENT ( THORIN_RUNTIME_PLATFORMS "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/platforms" ABSOLUTE )
SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES COMPILE_DEFINITIONS THORIN_RUNTIME_PLATFORMS=\"${THORIN_RUNTIME_PLATFORMS}/\")

IF ( BUILD_STATIC )
    MESSAGE ( "Building thorin as static library ..." )
    ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${SOURCES} )
ELSE ()
    MESSAGE ( "Building thorin as shared library ..." )
    ADD_LIBRARY ( ${LIBRARY_NAME} SHARED ${SOURCES} )
ENDIF ()

TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${THORIN_LIBS} )
