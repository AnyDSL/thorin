CMAKE_MINIMUM_REQUIRED(VERSION 2.8.1 FATAL_ERROR)

OPTION ( BUILD_IMPALA "Builds the Impala frontend." ON )

INCLUDE_DIRECTORIES ( ${PROJ_SOURCE_DIR} )
LINK_DIRECTORIES ( ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )

# enable c++11
IF (NOT MSVC)
    ADD_DEFINITIONS ( "-std=c++11" )
ENDIF ()

# check for wfv2
# TODO

# check for possible llvm extension
FIND_PACKAGE ( LLVM QUIET )

IF ( LLVM_FOUND )
    MESSAGE ( STATUS "Building with LLVM." )
    ADD_DEFINITIONS ( ${LLVM_DEFINITIONS} )
    ADD_DEFINITIONS ( "-DLLVM_SUPPORT" )
    INCLUDE_DIRECTORIES ( ${LLVM_INCLUDE_DIRS} )
    LINK_DIRECTORIES ( ${LLVM_LIBRARY_DIRS} )
ELSE ()
    MESSAGE ( STATUS "Building without LLVM. Specify LLVM_DIR to compile with LLVM." )
ENDIF ()

INCLUDE ( anydsl2-shared )

# build anydsl2 lib
ADD_SUBDIRECTORY ( anydsl2 )

#build impala frontend
IF ( BUILD_IMPALA )
    ADD_SUBDIRECTORY ( impala )
ENDIF ()

