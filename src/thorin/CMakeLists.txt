SET ( LIBRARY_NAME thorin )

# get source files
FILE ( GLOB_RECURSE SOURCES *.cpp *.h )

# create lib
get_thorin_dependency_libs( THORIN_LIBS )
ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${SOURCES} )
TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${THORIN_LIBS} )
