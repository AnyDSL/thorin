# Find the OpenCL includes and library
#
# To set manually the paths, define these environment variables:
#  OPENCL_INC_DIR       - Include path (e.g. OPENCL_INC_DIR=/usr/local/cuda/include)
#  OPENCL_LIB_DIR       - Library path (e.g. OPENCL_LIB_DIR=/usr/lib64/nvidia)
#
# Once done this will define
#  OPENCL_INCLUDE_DIRS  - where to find OpenCL include files
#  OPENCL_LIBRARY_DIRS  - where to find OpenCL libs
#  OPENCL_CFLAGS        - OpenCL C compiler flags
#  OPENCL_LFLAGS        - OpenCL linker flags
#  OPENCL_FOUND         - True if OpenCL is found

SET(OPENCL_INC_DIR $ENV{OPENCL_INC_DIR} CACHE PATH "OpenCL header files directory.")
SET(OPENCL_LIB_DIR $ENV{OPENCL_LIB_DIR} CACHE PATH "OpenCL library files directory.")

IF(APPLE)
    FIND_PATH(OPENCL_INCLUDE_DIR OpenCL/cl.h)
    FIND_LIBRARY(OPENCL_LIBRARY OpenCL)
    SET(OPENCL_CFLAGS "")
    SET(OPENCL_LFLAGS "-framework OpenCL")
ELSE(APPLE)
    # Unix style platforms
    FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h HINTS ${OPENCL_INC_DIR})
    FIND_LIBRARY(OPENCL_LIBRARY OpenCL HINTS ${OPENCL_LIB_DIR})
    GET_FILENAME_COMPONENT(OPENCL_LIBRARY_DIR ${OPENCL_LIBRARY} PATH)
    SET(OPENCL_CFLAGS "-I${OPENCL_INCLUDE_DIR}")
    SET(OPENCL_LFLAGS "-L${OPENCL_LIBRARY_DIR} -lOpenCL")
ENDIF(APPLE)

SET(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
SET(OPENCL_LIBRARY_DIRS ${OPENCL_LIBRARY_DIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OPENCL_INCLUDE_DIR OPENCL_LIBRARY)

MARK_AS_ADVANCED(OPENCL_INCLUDE_DIR OPENCL_LIBRARY_DIR OPENCL_LIBRARY)

