SET ( RUNTIME_DEFINITIONS "" )

# Look for CUDA
FIND_PACKAGE ( CUDA QUIET )
IF ( CUDA_FOUND )
    FIND_LIBRARY ( CUDA_NVVM_LIBRARY  nvvm  HINTS ${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib ${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib64)
    FIND_LIBRARY ( CUDA_NVRTC_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib      ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    INCLUDE_DIRECTORIES ( ${CUDA_INCLUDE_DIRS} "${CUDA_TOOLKIT_ROOT_DIR}/nvvm/include")
    SET ( CUDA_PLATFORM cuda_platform.cpp cuda_platform.h )
    SET ( RUNTIME_DEFINITIONS ${RUNTIME_DEFINITIONS} -DENABLE_CUDA )
ELSE ()
    SET ( CUDA_PLATFORM "" )
ENDIF ()

IF (NOT MSVC)
    ADD_DEFINITIONS ( "-std=c++11 -Wall -Wunused" )
ENDIF ()

ADD_DEFINITIONS ( ${RUNTIME_DEFINITIONS} )

ADD_LIBRARY ( thorin_runtime
              runtime.cpp
              runtime.h
              thorin_runtime.h
              platform.h
              cpu_platform.h
              dummy_platform.h
              ${CUDA_PLATFORM} )

