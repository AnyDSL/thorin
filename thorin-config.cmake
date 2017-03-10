# Try to find Thorin library and include path.
# Once done this will define
#
# Thorin_INCLUDE_DIRS
# Thorin_LIBRARIES (including dependencies to LLVM/RV)
# Thorin_FOUND

cmake_minimum_required(VERSION 3.1)

find_path(THORIN_ROOT_DIR thorin-config.cmake PATHS ${THORIN_DIR} $ENV{THORIN_DIR})
list(APPEND CMAKE_MODULE_PATH "${THORIN_ROOT_DIR}")
list(APPEND CMAKE_MODULE_PATH "${THORIN_ROOT_DIR}/cmake/modules")

find_package(LLVM QUIET)
find_package(RV QUIET)

function(generate_library_names OUT_VAR LIB)
    set(${OUT_VAR} ${LIB}.lib ${LIB}.so ${LIB}.a ${LIB}.dll ${LIB}.dylib lib${LIB} lib${LIB}.so lib${LIB}.a lib${LIB}.dll lib${LIB}.dylib PARENT_SCOPE)
endfunction()

generate_library_names(THORIN_OUTPUT_LIBS thorin)

find_path(THORIN_INCLUDE_DIR NAMES thorin/world.h PATHS ${THORIN_ROOT_DIR}/src)
find_path(THORIN_LIBS_DIR
    NAMES
        ${THORIN_OUTPUT_LIBS}
    PATHS
        ${THORIN_LIBS_DIR}
        ${THORIN_ROOT_DIR}/build_debug/lib
        ${THORIN_ROOT_DIR}/build_release/lib
        ${THORIN_ROOT_DIR}/build/lib
    PATH_SUFFIXES
        ${CMAKE_CONFIGURATION_TYPES}
)

# include AnyDSL specific stuff
include(${CMAKE_CURRENT_LIST_DIR}/thorin-shared.cmake)
find_library(THORIN_LIBRARY NAMES ${THORIN_OUTPUT_LIBS} PATHS ${THORIN_LIBS_DIR})
get_thorin_dependency_libs(THORIN_TEMP_LIBS)

set(Thorin_LIBRARIES ${THORIN_LIBRARY} ${THORIN_TEMP_LIBS})
set(Thorin_INCLUDE_DIRS ${THORIN_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thorin DEFAULT_MSG THORIN_LIBRARY THORIN_INCLUDE_DIR)

mark_as_advanced(THORIN_LIBRARY THORIN_INCLUDE_DIR THORIN_ROOT_DIR THORIN_LIBS_DIR)
