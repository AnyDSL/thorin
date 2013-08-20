SET ( LIBRARY_NAME anydsl2 )

# get source files
FILE ( GLOB CPP_GLOBAL    *.cpp *.h )
FILE ( GLOB CPP_ANALYSES  analyses/*.cpp analysis/*.h)
FILE ( GLOB CPP_TABLES    tables/*.cpp tables/*.h )
FILE ( GLOB CPP_TRANSFORM transform/*.cpp tansform/*.h )
FILE ( GLOB CPP_UTIL      util/*.cpp util/*.h )
FILE ( GLOB CPP_BACKEND   be/*.cpp be/*.h )

SET ( ANYDSL2_SOURCE_FILES ${CPP_GLOBAL} ${CPP_ANALYSES} ${CPP_TABLES} ${CPP_TRANSFORM} ${CPP_UTIL} ${CPP_BACKEND} )

# create lib
get_anydsl2_dependency_libs( ANYDSL2_LIBS )
ADD_LIBRARY ( ${LIBRARY_NAME} STATIC ${ANYDSL2_SOURCE_FILES} )
TARGET_LINK_LIBRARIES ( ${LIBRARY_NAME} ${ANYDSL2_LIBS} )
