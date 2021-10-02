cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)

project(Thorin)

include(CTest)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(THORIN_PROFILE "profile complexity in thorin::HashTable - only works in Debug build" ON)

set(PACKAGE_VERSION "0.4.9")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited config" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug or Release" FORCE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# check for possible llvm extension
find_package(LLVM CONFIG QUIET)
if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    # check for RV
    find_package(RV)
    if(RV_FOUND)
        message(STATUS "Building with RV from LLVM installation.")
    else()
        message(STATUS "Building without RV. Install RV as part of LLVM.")
    endif()
else()
    message(STATUS "Building without LLVM and RV. Specify LLVM_DIR to compile with LLVM.")
endif()

message(STATUS "Using Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Using Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
if(DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# configure thorin preprocessor definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(THORIN_ENABLE_CHECKS TRUE)
endif()
if(THORIN_PROFILE)
    set(THORIN_ENABLE_PROFILING TRUE)
endif()
if (RV_FOUND)
    set(THORIN_ENABLE_RV TRUE)
endif()
configure_file(src/thorin/config.h.in ${CMAKE_BINARY_DIR}/include/thorin/config.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(src)

if(BUILD_TESTING)
    include(GoogleTest)
    add_subdirectory(googletest)
    add_subdirectory(gtest)

    # keep cache clean
    mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
    )

    # keep IDEs folders clean
    set_target_properties(gtest      PROPERTIES FOLDER extern)
    set_target_properties(gtest_main PROPERTIES FOLDER extern)
    set_target_properties(gmock      PROPERTIES FOLDER extern)
    set_target_properties(gmock_main PROPERTIES FOLDER extern)
endif()

export(TARGETS libthorin FILE ${CMAKE_BINARY_DIR}/share/anydsl/cmake/thorin-exports.cmake)
configure_file(cmake/thorin-config.cmake.in ${CMAKE_BINARY_DIR}/share/anydsl/cmake/thorin-config.cmake @ONLY)
