if(MSVC)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4800 /wd4520")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4800 /wd4520")
else()
    add_definitions("-Wall -Wextra")
endif()

# check for Half library
find_package(Half REQUIRED)
message(STATUS "Building with Half library from ${Half_INCLUDE_DIRS}.")
include_directories(${Half_INCLUDE_DIRS})

# check for RV
find_package(RV QUIET)
if(RV_FOUND)
    message(STATUS "Building with RV.")
    add_definitions("-DRV_SUPPORT")
    include_directories(${RV_INCLUDE_DIR})
    link_directories(${RV_LIBRARY_DIR})
else()
    message(STATUS "Building without RV. Specify RV_DIR to compile with RV.")
endif()

# check for possible llvm extension
find_package(LLVM CONFIG QUIET)
if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    add_definitions(${LLVM_DEFINITIONS})
    add_definitions("-DLLVM_SUPPORT")
    include_directories(${LLVM_INCLUDE_DIRS})
else()
    message(STATUS "Building without LLVM. Specify LLVM_DIR to compile with LLVM.")
endif()

# check for colored terminal
SET(COLOR_TTY_AVAILABLE TRUE)
if(WIN32)
    # Windows console does not support ANSI escape codes
    set(COLOR_TTY_AVAILABLE FALSE)
endif()
set(COLORIZE_OUTPUT ${COLOR_TTY_AVAILABLE} CACHE BOOL "Set to true to enable colored error messages")
if(COLORIZE_OUTPUT)
    add_definitions(-DCOLORIZE_LOG)
endif()

# build thorin lib
include(../thorin-shared.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(thorin)
