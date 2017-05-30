# Find the RV library
#
# Once done this will define
#  RV_INCLUDE_DIRS  - where to find RV library include file
#  RV_LIBRARIES     - where to find RV library
#  RV_FOUND         - True if RV library is found

find_path(RV_INCLUDE_DIR      rv.h     PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES rv)
find_library(RV_LIBRARY       RV       PATHS ${LLVM_LIBRARY_DIRS})
find_library(RV_SLEEF_LIBRARY gensleef PATHS ${LLVM_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RV DEFAULT_MSG RV_INCLUDE_DIR RV_LIBRARY)

set(RV_INCLUDE_DIRS ${RV_INCLUDE_DIR})
set(RV_LIBRARIES    ${RV_LIBRARY})
if(RV_SLEEF_LIBRARY)
    list(APPEND RV_LIBRARIES ${RV_SLEEF_LIBRARY})
endif()

mark_as_advanced(RV_INCLUDE_DIR RV_LIBRARY RV_SLEEF_LIBRARY)
