SET ( LIBRARY_NAME thorin )

# get source files
FILE ( GLOB_RECURSE SOURCES *.cpp *.h )

# create lib
get_thorin_dependency_libs( THORIN_LIBS )

IF ( BUILD_STATIC )
    MESSAGE ( "Building thorin as static library ..." )
    ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${SOURCES} )
ELSE ()
    MESSAGE ( "Building thorin as shared library ..." )
    ADD_LIBRARY ( ${LIBRARY_NAME} SHARED ${SOURCES} )
ENDIF ()

TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${THORIN_LIBS} )
