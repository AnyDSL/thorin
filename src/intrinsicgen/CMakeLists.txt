SET ( EXECUTABLE_NAME intrinsicgen )

# get source files
FILE ( GLOB SOURCES *.cpp *.h )

# create executable
get_thorin_dependency_libs( THORIN_LIBS )
ADD_EXECUTABLE( ${EXECUTABLE_NAME} ${SOURCES} )
TARGET_LINK_LIBRARIES ( ${EXECUTABLE_NAME} ${THORIN_LIBS} lib_impala )
ADD_DEPENDENCIES ( ${EXECUTABLE_NAME} lib_impala thorin )
