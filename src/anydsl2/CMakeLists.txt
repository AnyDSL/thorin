SET ( LIBRARY_NAME anydsl2 )

# get source files
FILE ( GLOB CPP_GLOBAL    *.cpp )
FILE ( GLOB CPP_ANALYSES  analyses/*.cpp )
FILE ( GLOB CPP_TABLES    tables/*.cpp )
FILE ( GLOB CPP_TRANSFORM transform/*.cpp )
FILE ( GLOB CPP_UTIL      util/*.cpp )

# backend source files
SET ( CPP_BE_LLVM          be/llvm.cpp )

# list of source files (without backends)
SET ( ANYDSL2_SOURCE_FILES ${CPP_GLOBAL} ${CPP_ANALYSES} ${CPP_TABLES} ${CPP_TRANSFORM} ${CPP_UTIL} )

# enable LLVM backend (if possible)
IF ( LLVM_FOUND )
    SET ( ANYDSL2_SOURCE_FILES ${ANYDSL2_SOURCE_FILES} ${CPP_BE_LLVM} )
ENDIF ()

# create lib
get_anydsl2_dependency_libs( ANYDSL2_LIBS )
ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${ANYDSL2_SOURCE_FILES} )
TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${ANYDSL2_LIBS} )
