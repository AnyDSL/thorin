include(CMakeParseArguments)

macro(THORIN_RUNTIME_WRAP outfiles outlibs)
	CMAKE_PARSE_ARGUMENTS("TRW" "MAIN" "RTTYPE" "FILES" ${ARGN})
	IF(NOT "${TRW_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(FATAL_ERROR "Unparsed arguments ${TRW_UNPARSED_ARGUMENTS}")
	ENDIF()
	# add the common runtime
	set(_impala_platform ${THORIN_RUNTIME_DIR}/platforms/intrinsics_thorin.impala)
	set(${outfiles} ${THORIN_RUNTIME_DIR}/common/thorin_runtime.cpp)
	IF("${TRW_MAIN}")
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/common/thorin_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "-DPROVIDE_MAIN"
		)
	ENDIF()
	# add specific runtime
	set(CUDA_RUNTIME_DEFINES "'-DLIBDEVICE_DIR=\"${CUDA_DIR}/nvvm/libdevice/\"' '-DNVCC_BIN=\"${CUDA_DIR}/bin/nvcc\"' '-DKERNEL_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\"'")
	set(CUDA_RUNTIME_INCLUDES "-I${CUDA_DIR}/include -I${CUDA_DIR}/nvvm/include -I${CUDA_DIR}/nvvm/libnvvm-samples/common/include")
	IF("${TRW_RTTYPE}" STREQUAL "nvvm")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp)
		set(${outlibs} cuda ${CUDA_DIR}/nvvm/lib64/libnvvm.so)
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_nvvm.impala)
		# lucky enough, cmake does the right thing here even when we compile impala programs from various folders
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "${CUDA_RUNTIME_DEFINES} ${CUDA_RUNTIME_INCLUDES}"
		)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "cuda")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp)
		set(${outlibs} cuda ${CUDA_DIR}/nvvm/lib64/libnvvm.so)
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_cuda.impala)
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "${CUDA_RUNTIME_DEFINES} ${CUDA_RUNTIME_INCLUDES}"
		)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "spir")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/opencl/cl_runtime.cpp)
		FIND_LIBRARY(CL_LIB OpenCL)
		set(${outlibs} ${CL_LIB})
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_spir.impala)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "opencl")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/opencl/cl_runtime.cpp)
		FIND_LIBRARY(CL_LIB OpenCL)
		set(${outlibs} ${CL_LIB})
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_opencl.impala)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "cpu")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cpu/cpu_runtime.cpp)
		set(${outlibs})
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_cpu.impala)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "avx")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cpu/cpu_runtime.cpp)
		set(${outlibs})
		set(_impala_platform ${_impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_avx.impala)
	ELSE()
		message(FATAL_ERROR "Unknown runtime type ${TRW_RTTYPE}")
	ENDIF()
	# get the options right
	set(_clangopts ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
	separate_arguments(_clangopts)
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
	# prepare platform symlinks in build directory
	execute_process(COMMAND ln -fs ${THORIN_RUNTIME_DIR}/platforms/generic.s ${THORIN_RUNTIME_DIR}/platforms/nvvm.s ${THORIN_RUNTIME_DIR}/platforms/spir.s ${CMAKE_CURRENT_BINARY_DIR})
	# tell cmake what to do
	add_custom_command(OUTPUT ${_llfile}
		COMMAND impala
		ARGS ${_impala_platform} ${_infiles} -emit-llvm -O2
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		DEPENDS ${_impala_platform} ${_infiles} VERBATIM)
	add_custom_command(OUTPUT ${_objfile}
		COMMAND clang++
		ARGS ${_clangopts} -g -c -o ${_objfile} ${_llfile}
		DEPENDS ${_llfile} VERBATIM)
	SET_SOURCE_FILES_PROPERTIES(
		${_objfile}
		PROPERTIES
		EXTERNAL_OBJECT true
		GENERATED true)
	set(${outfiles} ${${outfiles}} ${_objfile})
endmacro()
