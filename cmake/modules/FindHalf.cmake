# Find the Half library
#
# To set manually the paths, define these environment variables:
#  Half_DIR       - Include path
#
# Once done this will define
#  Half_INCLUDE_DIRS  - where to find Half library include file
#  Half_FOUND         - True if Half library is found

SET(Half_DIR $ENV{Half_DIR} CACHE PATH "C++ library for half precision floating point arithmetics.")
FIND_PATH(Half_INCLUDE_DIR half.hpp PATHS ${Half_DIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Half DEFAULT_MSG Half_INCLUDE_DIR)

SET(Half_INCLUDE_DIRS ${Half_INCLUDE_DIR})

MARK_AS_ADVANCED(Half_INCLUDE_DIR)
