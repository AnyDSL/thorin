# Find the Half library
#
# To set manually the paths, define these environment variables:
#  HALF_DIR       - Include path
#
# Once done this will define
#  HALF_INCLUDE_DIRS  - where to find Half library include file
#  HALF_FOUND         - True if Half library is found

SET(Half_DIR $ENV{Half_DIR} CACHE PATH "C++ library for half precision floating point arithmetics.")
FIND_PATH(HALF_INCLUDE_DIR half.hpp PATHS ${Half_DIR})

SET(HALF_INCLUDE_DIRS ${HALF_INCLUDE_DIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HALF DEFAULT_MSG HALF_INCLUDE_DIR)

MARK_AS_ADVANCED(HALF_INCLUDE_DIR)
