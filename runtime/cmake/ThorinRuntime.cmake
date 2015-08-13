include(CMakeParseArguments)

# find impala
find_program(IMPALA_BIN impala)
IF(NOT IMPALA_BIN)
    message(FATAL_ERROR "Could not find impala binary, it has to be in the PATH")
ENDIF()
find_program(LLVM_AS_BIN llvm-as)
find_program(CLANGPP_BIN clang++)
IF(NOT LLVM_AS_BIN)
    message(FATAL_ERROR "Could not find llvm-as binary, it has to be in the PATH")
ENDIF()
IF(NOT CLANGPP_BIN)
    message(FATAL_ERROR "Could not find clang++ binary, it has to be in the PATH")
ENDIF()

# find python for post-patcher.py
find_package(PythonInterp REQUIRED)
message(STATUS "Python found: ${PYTHON_VERSION_STRING}")
set(PYTHON_BIN ${PYTHON_EXECUTABLE})

SET(BACKEND ${BACKEND} CACHE STRING "select the backend from the following: CPU, AVX, NVVM, CUDA, OPENCL, SPIR")
IF(NOT BACKEND)
    SET(BACKEND cpu CACHE STRING "select the backend from the following: CPU, AVX, NVVM, CUDA, OPENCL, SPIR" FORCE)
ENDIF()
STRING(TOLOWER "${BACKEND}" BACKEND)
MESSAGE(STATUS "Selected backend: ${BACKEND}")

macro(THORIN_RUNTIME_WRAP outfiles outlibs)
    CMAKE_PARSE_ARGUMENTS("TRW" "MAIN" "BACKEND" "FILES" ${ARGN})
    IF(NOT "${TRW_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR "Unparsed arguments ${TRW_UNPARSED_ARGUMENTS}")
    ENDIF()

    ## add runtime
    # add the common runtime
    set(_impala_platform ${THORIN_RUNTIME_DIR}/platforms/intrinsics_thorin.impala ${THORIN_RUNTIME_DIR}/platforms/intrinsics_utils.impala)
    set(${outfiles} ${THORIN_RUNTIME_DIR}/common/thorin_runtime.cpp ${THORIN_RUNTIME_DIR}/common/thorin_utils.cpp)
    set(${outlibs})
    IF("${TRW_MAIN}")
        SET_SOURCE_FILES_PROPERTIES(
            ${THORIN_RUNTIME_DIR}/common/thorin_runtime.cpp
            PROPERTIES
            COMPILE_FLAGS "-DPROVIDE_MAIN"
        )
    ENDIF()
    # add specific runtime
    IF("${TRW_BACKEND}" STREQUAL "nvvm" OR "${TRW_BACKEND}" STREQUAL "cuda")
        Find_Package(CUDA REQUIRED)
        set(CUDA_RUNTIME_DEFINES "'-DLIBDEVICE_DIR=\"${CUDA_TOOLKIT_ROOT_DIR}/nvvm/libdevice/\"' '-DNVCC_BIN=\"${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc\"' '-DKERNEL_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\"'")
        set(CUDA_RUNTIME_INCLUDES "-I${CUDA_INCLUDE_DIRS} -I${CUDA_TOOLKIT_ROOT_DIR}/nvvm/include")
        # set variables expected below
        set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp)
        Find_Library(CUDA_NVVM_LIBRARY nvvm HINTS ${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib ${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib64)
        set(${outlibs} ${${outlibs}} ${CUDA_CUDA_LIBRARY} ${CUDA_NVVM_LIBRARY})
        IF(NOT (CUDA_VERSION VERSION_LESS "7.00"))
            Find_Library(CUDA_NVRTC_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
            set(${outlibs} ${${outlibs}} ${CUDA_NVRTC_LIBRARY})
        ENDIF()
        set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_${TRW_BACKEND}.impala)
        # cu_runtime needs some defines
        # lucky enough, cmake does the right thing here even when we compile impala programs from various folders
        SET_SOURCE_FILES_PROPERTIES(
            ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp
            PROPERTIES
            COMPILE_FLAGS "${CUDA_RUNTIME_DEFINES} ${CUDA_RUNTIME_INCLUDES}"
        )
    ELSEIF("${TRW_BACKEND}" STREQUAL "spir" OR "${TRW_BACKEND}" STREQUAL "opencl")
        FIND_LIBRARY(CL_LIB OpenCL ENV CL_LIB)
        IF(APPLE)
            FIND_PATH(CL_INC OpenCL/cl.h)
        ELSE()
            FIND_PATH(CL_INC CL/cl.h)
        ENDIF()
        include_directories(${CL_INC})
        # set variables expected below
        set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/opencl/cl_runtime.cpp)
        set(${outlibs} ${${outlibs}} ${CL_LIB})
        set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_${TRW_BACKEND}.impala)
        # cl_runtime needs some defines
        # lucky enough, cmake does the right thing here even when we compile impala programs from various folders
        SET_SOURCE_FILES_PROPERTIES(
            ${THORIN_RUNTIME_DIR}/opencl/cl_runtime.cpp
            PROPERTIES
            COMPILE_FLAGS "'-DKERNEL_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\"'"
        )
    ELSEIF("${TRW_BACKEND}" STREQUAL "cpu" OR "${TRW_BACKEND}" STREQUAL "avx")
        ENABLE_LANGUAGE(C)
        find_package(Threads REQUIRED)
        # find tbb package
        Find_Path(TBB_INCLUDE_DIRS tbb/tbb.h HINTS ${TBB_INC_DIR})
        Find_Library(TBB_LIBRARY tbb HINTS ${TBB_LIB_DIR})
        Get_Filename_Component(TBB_LIBRARY_DIR ${TBB_LIBRARY} PATH)
        Find_Package_Handle_Standard_Args(TBB DEFAULT_MSG TBB_INCLUDE_DIRS TBB_LIBRARY)
        # set variables expected below
        set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cpu/cpu_runtime.cpp)
        IF(TBB_FOUND)
            INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
            set(${outlibs} ${${outlibs}} ${TBB_LIBRARY})
            ADD_DEFINITIONS(-DUSE_TBB)
        ELSE()
            set(${outlibs} ${${outlibs}} ${CMAKE_THREAD_LIBS_INIT})
            IF(NOT MSVC)
                set(${outlibs} ${${outlibs}} -pthread)
                ADD_DEFINITIONS(-pthread)
            ENDIF()
        ENDIF()
        set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_${TRW_BACKEND}.impala)
    ELSE()
        message(FATAL_ERROR "Unknown backend: ${TRW_BACKEND}")
    ENDIF()

    ## generate files
    # get the options right
    # get last filename, and absolute filenames
    set(_infiles)
    foreach(_it ${TRW_FILES})
        get_filename_component(_infile ${_it} ABSOLUTE)
        set(_infiles ${_infiles} ${_infile})
        set(_lastfile ${_it})
    endforeach()
    # add all input files as one impala job
    get_filename_component(_basename ${_lastfile} NAME_WE)
    set(_llfile ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.ll)
    set(_objfile ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.o)
    # tell cmake what to do
    add_custom_command(OUTPUT ${_llfile}
        COMMAND ${IMPALA_BIN} ${_impala_platform} ${_infiles} -emit-llvm -O3
        COMMAND ${PYTHON_BIN} ${THORIN_RUNTIME_DIR}/post-patcher.py ${TRW_BACKEND} ${CMAKE_CURRENT_BINARY_DIR}/${_basename}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IMPALA_BIN} ${THORIN_LIBRARY} ${PYTHON_BIN} ${THORIN_RUNTIME_DIR}/post-patcher.py ${_impala_platform} ${_infiles} VERBATIM)
    IF("${TRW_BACKEND}" STREQUAL "spir")
        set(_spirfile ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.spir)
        set(_bcfile ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.spir.bc)
        add_custom_command(OUTPUT ${_bcfile}
            COMMAND ${LLVM_AS_BIN} ${_spirfile}
            DEPENDS ${_spirfile} VERBATIM)
    ENDIF()
    add_custom_command(OUTPUT ${_objfile}
        COMMAND ${CLANGPP_BIN} -O3 -march=native -g -c -o ${_objfile} ${_llfile}
        DEPENDS ${_llfile} VERBATIM)
    SET_SOURCE_FILES_PROPERTIES(
        ${_objfile}
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true)
    set(${outfiles} ${${outfiles}} ${_objfile} ${_bcfile})
endmacro()
