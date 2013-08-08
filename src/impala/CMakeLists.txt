SET ( EXECUTABLE_NAME impala )

# get source files
FILE ( GLOB CPP_GLOBAL *.cpp )

# list of source files (without backends)
SET ( IMPALA_SOURCE_FILES ${CPP_GLOBAL} )

# setup linking libs
SET ( IMPALA_LIBS )

# create executable
get_anydsl2_dependency_libs( ANYDSL2_LIBS )
ADD_EXECUTABLE( ${EXECUTABLE_NAME} ${IMPALA_SOURCE_FILES} )
TARGET_LINK_LIBRARIES ( ${EXECUTABLE_NAME} ${ANYDSL2_LIBS} ${IMPALA_LIBS} anydsl2 )
ADD_DEPENDENCIES ( ${EXECUTABLE_NAME} anydsl2 )
