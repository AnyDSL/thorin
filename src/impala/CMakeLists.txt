SET ( EXECUTABLE_NAME impala )

# get source files
FILE ( GLOB CPP_GLOBAL *.cpp *.h )

# list of source files (without backends)
SET ( IMPALA_SOURCE_FILES ${CPP_GLOBAL} )

# setup linking libs
SET ( IMPALA_LIBS )

# create executable
get_thorin_dependency_libs( THORIN_LIBS )
ADD_EXECUTABLE( ${EXECUTABLE_NAME} ${IMPALA_SOURCE_FILES} )
TARGET_LINK_LIBRARIES ( ${EXECUTABLE_NAME} ${THORIN_LIBS} ${IMPALA_LIBS} thorin )
ADD_DEPENDENCIES ( ${EXECUTABLE_NAME} thorin )
