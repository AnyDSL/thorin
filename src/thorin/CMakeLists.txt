SET ( LIBRARY_NAME thorin )

# get source files
FILE ( GLOB CPP_GLOBAL    *.cpp *.h )
FILE ( GLOB CPP_ANALYSES  analyses/*.cpp analyses/*.h)
FILE ( GLOB CPP_TABLES    tables/*.cpp tables/*.h )
FILE ( GLOB CPP_TRANSFORM transform/*.cpp transform/*.h )
FILE ( GLOB CPP_UTIL      util/*.cpp util/*.h )
FILE ( GLOB CPP_BACKEND   be/*.cpp be/*.h )

SET ( THORIN_SOURCE_FILES ${CPP_GLOBAL} ${CPP_ANALYSES} ${CPP_TABLES} ${CPP_TRANSFORM} ${CPP_UTIL} ${CPP_BACKEND} )

# create lib
get_thorin_dependency_libs( THORIN_LIBS )
ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${THORIN_SOURCE_FILES} )
TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${THORIN_LIBS} )
