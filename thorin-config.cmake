# Try to find Thorin library and include path.
# Once done this will define
#
# THORIN_FOUND
# THORIN_INCLUDE_DIR
# THORIN_LIBS (including dependencies to LLVM/WFV2)
# THORIN_LIBS_DIR
# THORIN_RUNTIME_DIR

SET ( PROJ_NAME THORIN )

FIND_PACKAGE ( LLVM QUIET )
FIND_PACKAGE ( WFV2 QUIET )

FIND_PATH ( THORIN_ROOT_DIR thorin-config.cmake PATHS ${THORIN_DIR} $ENV{THORIN_DIR} )
SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${THORIN_ROOT_DIR} )

SET ( THORIN_OUTPUT_LIBS thorin.lib thorin.so thorin.a thorin.dll thorin.dylib libthorin libthorin.so libthorin.a libthorin.dll libthorin.dylib )

FIND_PATH ( THORIN_INCLUDE_DIR NAMES thorin/world.h PATHS ${THORIN_ROOT_DIR}/src )
FIND_PATH ( THORIN_LIBS_DIR
    NAMES
        ${THORIN_OUTPUT_LIBS}
    PATHS
        ${THORIN_ROOT_DIR}/build_debug/lib
        ${THORIN_ROOT_DIR}/build_release/lib
        ${THORIN_ROOT_DIR}/build/lib
    PATH_SUFFIXES
        ${CMAKE_CONFIGURATION_TYPES}
)
FIND_PATH ( THORIN_RUNTIME_DIR
    NAMES
        cmake/ThorinRuntime.cmake platforms/intrinsics_thorin.impala
    PATHS
        ${THORIN_ROOT_DIR}/runtime
)
FIND_PATH ( THORIN_RUNTIME_INCLUDE_DIR
    NAMES
        thorin_runtime.h
    PATHS
        ${THORIN_ROOT_DIR}/runtime/common
)

# include AnyDSL specific stuff
INCLUDE ( ${CMAKE_CURRENT_LIST_DIR}/thorin-shared.cmake )

IF ( THORIN_LIBS_DIR )
    FIND_LIBRARY ( THORIN_LIBS_DEBUG NAMES ${THORIN_OUTPUT_LIBS} PATHS ${THORIN_LIBS_DIR} PATH_SUFFIXES Debug)
    FIND_LIBRARY ( THORIN_LIBS_RELEASE NAMES ${THORIN_OUTPUT_LIBS} PATHS ${THORIN_LIBS_DIR} PATH_SUFFIXES Release )
    SET ( THORIN_LIBS
        optimized ${THORIN_LIBS_RELEASE}
        debug ${THORIN_LIBS_DEBUG}
    )
    get_thorin_dependency_libs ( THORIN_TEMP_LIBS )
    SET ( THORIN_LIBS ${THORIN_TEMP_LIBS} ${THORIN_LIBS} )
ENDIF()

IF ( THORIN_INCLUDE_DIR AND THORIN_LIBS )
    SET ( THORIN_FOUND TRUE CACHE BOOL "" FORCE )
ELSE()
    SET ( THORIN_FOUND FALSE CACHE BOOL "" FORCE )
ENDIF()

MARK_AS_ADVANCED ( THORIN_FOUND )
