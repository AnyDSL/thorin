# Find the OpenCL includes and library
#
# To set manually the paths, define these environment variables:
#  OpenCL_INC_DIR       - Include path (e.g. OpenCL_INC_DIR=/usr/local/cuda/include)
#  OpenCL_LIB_DIR       - Library path (e.g. OpenCL_LIB_DIR=/usr/lib64/nvidia)
#
# Once done this will define
#  OpenCL_INCLUDE_DIRS  - where to find OpenCL include files
#  OpenCL_LIBRARIES     - where to find OpenCL libs
#  OpenCL_FOUND         - True if OpenCL is found

SET(OpenCL_INC_DIR $ENV{OpenCL_INC_DIR} CACHE PATH "OpenCL header files directory.")
SET(OpenCL_LIB_DIR $ENV{OpenCL_LIB_DIR} CACHE PATH "OpenCL library files directory.")

IF(APPLE)
    FIND_PATH(OpenCL_INCLUDE_DIR OpenCL/cl.h)
    FIND_LIBRARY(OpenCL_LIBRARY OpenCL)
ELSE(APPLE)
    FIND_PATH(OpenCL_INCLUDE_DIR CL/cl.h HINTS ${OpenCL_INC_DIR})
    FIND_LIBRARY(OpenCL_LIBRARY OpenCL HINTS ${OpenCL_LIB_DIR})
ENDIF(APPLE)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OpenCL_INCLUDE_DIR OpenCL_LIBRARY)

SET(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
SET(OpenCL_LIBRARIES ${OpenCL_LIBRARY})

MARK_AS_ADVANCED(OpenCL_INCLUDE_DIR OpenCL_LIBRARY)
