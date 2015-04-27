CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5 FATAL_ERROR)

PROJECT(CUDA_RUNTIME CXX)

FIND_PACKAGE(THORIN REQUIRED)
include(${THORIN_RUNTIME_DIR}/cmake/ThorinRuntime.cmake)
include_directories(${THORIN_RUNTIME_INCLUDE_DIR})

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(NOT MSVC)

THORIN_RUNTIME_WRAP(CU_TEST_FILES CU_TEST_LIBS
    BACKEND nvvm
    MAIN
    FILES cu_test.impala)
SET_SOURCE_FILES_PROPERTIES(
    cu_test_gen.cpp
    PROPERTIES
    COMPILE_FLAGS ${CUDA_RUNTIME_INCLUDES})
add_executable(cu_test cu_test_gen.cpp ${CU_TEST_FILES})
target_link_libraries(cu_test ${CU_TEST_LIBS})
add_test("cu_test" cu_test)
configure_file(cu_test.cu   ${CMAKE_CURRENT_BINARY_DIR}/cu_test.cu   COPYONLY)
configure_file(cu_test.nvvm ${CMAKE_CURRENT_BINARY_DIR}/cu_test.nvvm COPYONLY)
