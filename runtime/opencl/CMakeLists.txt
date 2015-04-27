CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5 FATAL_ERROR)

PROJECT(OPENCL_RUNTIME CXX)

FIND_PACKAGE(THORIN REQUIRED)
include(${THORIN_RUNTIME_DIR}/cmake/ThorinRuntime.cmake)
include_directories(${THORIN_RUNTIME_INCLUDE_DIR})

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(NOT MSVC)

THORIN_RUNTIME_WRAP(CL_TEST_FILES CL_TEST_LIBS
    BACKEND spir
    MAIN
    FILES cl_test.impala)
add_executable(cl_test cl_test_gen.cpp ${CL_TEST_FILES})
target_link_libraries(cl_test ${CL_TEST_LIBS})
add_test("cl_test" cl_test)
configure_file(cl_test.spir ${CMAKE_CURRENT_BINARY_DIR}/cl_test.spir COPYONLY)
