# Find the RV library
#
# Once done this will define
#  RV_INCLUDE_DIRS  - where to find RV library include file
#  RV_LIBRARIES     - where to find RV library
#  RV_FOUND         - True if RV library is found

find_path(RV_INCLUDE_DIR rv/rv.h
    PATHS
        ${LLVM_INCLUDE_DIRS}
        ${LLVM_EXTERNAL_RV_SOURCE_DIR}/include
        ${LLVM_BUILD_MAIN_SRC_DIR}/../rv/include
        ${LLVM_BUILD_MAIN_SRC_DIR}/tools/rv/include)
if(TARGET RV)
    set(RV_LIBRARY RV)
else()
    find_library(RV_LIBRARY       RV       PATHS ${LLVM_LIBRARY_DIRS})
endif()
if(TARGET gensleef)
    set(RV_SLEEF_LIBRARY gensleef)
else()
    find_library(RV_SLEEF_LIBRARY gensleef PATHS ${LLVM_LIBRARY_DIRS})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RV DEFAULT_MSG RV_INCLUDE_DIR RV_LIBRARY)

set(RV_INCLUDE_DIRS ${RV_INCLUDE_DIR})
set(RV_LIBRARIES    ${RV_LIBRARY})
if(RV_SLEEF_LIBRARY)
    list(APPEND RV_LIBRARIES ${RV_SLEEF_LIBRARY})
endif()

mark_as_advanced(RV_INCLUDE_DIR RV_LIBRARY RV_SLEEF_LIBRARY)
