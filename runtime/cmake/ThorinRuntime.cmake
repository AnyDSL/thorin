include(CMakeParseArguments)

set(CUDA_RUNTIME_DEFINES "'-DLIBDEVICE_DIR=\"${CUDA_DIR}/nvvm/libdevice/\"' '-DKERNEL_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\"'")
set(CUDA_RUNTIME_INCLUDES "-I${CUDA_DIR}/include -I${CUDA_DIR}/nvvm/include -I${CUDA_DIR}/nvvm/libnvvm-samples/common/include")

macro(THORIN_RUNTIME_WRAP outfiles outlibs)
	CMAKE_PARSE_ARGUMENTS("TRW" "MAIN" "RTTYPE" "FILES" ${ARGN})
	IF(NOT "${TRW_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(FATAL_ERROR "Unparsed arguments ${TRW_UNPARSED_ARGUMENTS}")
	ENDIF()
	# add the runtime
	set(impala_platform ${THORIN_RUNTIME_DIR}/platforms/intrinsics_thorin.impala)
	IF("${TRW_MAIN}")
		set(impala_defines "-DPROVIDE_MAIN")
	ELSE()
		set(impala_defines "")
	ENDIF()
	IF("${TRW_RTTYPE}" STREQUAL "nvvm")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp)
		set(${outlibs} ${${outlibs}} cuda ${CUDA_DIR}/nvvm/lib64/libnvvm.so)
		set(impala_platform ${impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_nvvm.impala)
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "${CUDA_RUNTIME_DEFINES} ${impala_defines} ${CUDA_RUNTIME_INCLUDES}"
		)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "cuda")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp)
		set(${outlibs} ${${outlibs}} cuda ${CUDA_DIR}/nvvm/lib64/libnvvm.so)
		set(impala_platform ${impala_platform} ${THORIN_RUNTIME_DIR}/platforms/intrinsics_cuda.impala)
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/cuda/cu_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "${CUDA_RUNTIME_DEFINES} ${impala_defines} ${CUDA_RUNTIME_INCLUDES}"
		)
	ELSEIF("${TRW_RTTYPE}" STREQUAL "plain")
		set(${outfiles} ${${outfiles}} ${THORIN_RUNTIME_DIR}/plain/plain_runtime.cpp)
		set(${outlibs} ${${outlibs}})
		SET_SOURCE_FILES_PROPERTIES(
			${THORIN_RUNTIME_DIR}/plain/plain_runtime.cpp
			PROPERTIES
			COMPILE_FLAGS "${impala_defines}"
		)
	ELSE()
		message(FATAL_ERROR "Unknown runtime type ${TRW_RTTYPE}")
	ENDIF()
	# get the options right
	set(CLANG_OPTS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
	separate_arguments(CLANG_OPTS)
	# go over input files
	foreach (it ${TRW_FILES})
		# get name of files
		get_filename_component(infile ${it} ABSOLUTE)
		get_filename_component(basefile ${it} NAME_WE)
		set(llfile ${CMAKE_CURRENT_BINARY_DIR}/${basefile}.ll)
		set(objfile ${CMAKE_CURRENT_BINARY_DIR}/${basefile}.o)
		# tell cmake what to do
		add_custom_command(OUTPUT ${llfile}
			COMMAND impala
			ARGS ${infile} -f -emit-llvm
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			DEPENDS ${infile} VERBATIM)
		add_custom_command(OUTPUT ${objfile}
			COMMAND clang++
			ARGS ${CLANG_OPTS} -g -c -o ${objfile} ${llfile}
			DEPENDS ${llfile} VERBATIM)
		SET_SOURCE_FILES_PROPERTIES(
			${objfile}
			PROPERTIES
			EXTERNAL_OBJECT true
			GENERATED true)
		set(${outfiles} ${${outfiles}} ${objfile})
	endforeach()
endmacro()
