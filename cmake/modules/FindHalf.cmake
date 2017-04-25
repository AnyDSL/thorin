# Find the Half library
#
# To set manually the paths, define these environment variables:
#  Half_DIR           - Include path
#
# Once done this will define
#  Half_INCLUDE_DIRS  - where to find Half library include file
#  Half_FOUND         - True if Half library is found

find_path(Half_DIR half.hpp PATHS ${Half_DIR} $ENV{Half_DIR} PATH_SUFFIXES include DOC "C++ library for half precision floating point arithmetics.")
find_path(Half_INCLUDE_DIR half.hpp PATHS ${Half_DIR} PATH_SUFFIXES include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Half DEFAULT_MSG Half_INCLUDE_DIR)

set(Half_INCLUDE_DIRS ${Half_INCLUDE_DIR})

mark_as_advanced(Half_INCLUDE_DIR)
