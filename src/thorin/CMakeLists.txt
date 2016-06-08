SET ( LIBRARY_NAME thorin )

# get source files
SET ( SOURCES
    continuation.cpp
    continuation.h
    def.cpp
    def.h
    enums.cpp
    enums.h
    henk.cpp.h
    henk.h
    irbuilder.cpp
    irbuilder.h
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
    transform/hoist_enters.cpp
    transform/hoist_enters.h
    transform/higher_order_lifting.cpp
    transform/higher_order_lifting.h
    transform/import.cpp
    transform/import.h
    transform/inliner.cpp
    transform/inliner.h
    transform/lift_builtins.cpp
    transform/lift_builtins.h
    transform/lower2cff.cpp
    transform/lower2cff.h
    transform/mangle.cpp
    transform/mangle.h
    transform/mem2reg.cpp
    transform/mem2reg.h
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
    util/stream.cpp
    util/stream.h
    util/types.h
    util/ycomp.cpp
    util/ycomp.h
)

# create lib
GET_THORIN_DEPENDENCY_LIBS ( THORIN_LIBS )

# set define for *.s file
GET_FILENAME_COMPONENT ( THORIN_RUNTIME_PLATFORMS "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/platforms" ABSOLUTE )
SET_SOURCE_FILES_PROPERTIES(be/llvm/runtime.cpp PROPERTIES COMPILE_DEFINITIONS THORIN_RUNTIME_PLATFORMS=\"${THORIN_RUNTIME_PLATFORMS}/\")

ADD_LIBRARY ( ${LIBRARY_NAME} ${SOURCES} )
TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${THORIN_LIBS} )
