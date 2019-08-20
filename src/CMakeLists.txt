if(MSVC)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options("/wd4800" "/wd4520")
else()
    add_definitions("-Wall -Wextra")
endif()

# check for Half library
find_package(Half REQUIRED)
message(STATUS "Building with Half library from ${Half_INCLUDE_DIRS}.")
include_directories(${Half_INCLUDE_DIRS})

if(LLVM_FOUND)
    add_definitions(${LLVM_DEFINITIONS})
    add_definitions("-DLLVM_SUPPORT")
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
endif()

if(RV_FOUND)
    include_directories(SYSTEM ${RV_INCLUDE_DIRS})
endif()

# check for colored terminal
set(COLOR_TTY_AVAILABLE TRUE)
if(WIN32)
    # Windows console does not support ANSI escape codes
    set(COLOR_TTY_AVAILABLE FALSE)
endif()
set(COLORIZE_OUTPUT ${COLOR_TTY_AVAILABLE} CACHE BOOL "Set to true to enable colored error messages")
if(COLORIZE_OUTPUT)
    add_definitions(-DCOLORIZE_LOG)
endif()

# build thorin lib
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(thorin)
