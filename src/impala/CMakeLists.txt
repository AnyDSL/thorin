
# Boost

SET(Boost_USE_STATIC_LIBS   OFF)
SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE( Boost 1.49.0 REQUIRED )

# LLVM

FIND_PACKAGE(LLVM REQUIRED)

IF( NOT LLVM_FOUND )
  MESSAGE(FATAL_ERROR "LLVM package can't be found. Set CMAKE_PREFIX_PATH variable to LLVM's installation prefix.")
ENDIF( NOT LLVM_FOUND )


# project sources

SET(LIBRARY_NAME
	impala
)

SET(LIBRARY_DEPENDENCIES
	anydsl2
)

SET(LIBRARY_EXT_INCLUDES
	${Boost_INCLUDE_DIR}
	${LLVM_INCLUDE_DIRS}
)


ADD_DEFINITIONS(${LLVM_DEFINITIONS})

llvm_map_components_to_libraries(REQ_LLVM_LIBRARIES jit native analysis)

LINK_DIRECTORIES(
	${LLVM_LIBRARY_DIRS}
)

SET(LIBRARY_EXT_LINK_LIBS
	${Boost_LIBRARIES}
	${REQ_LLVM_LIBRARIES}
)

add_project_static_library(
	"${LIBRARY_NAME}"
	"automatic"
	"${LIBRARY_DEPENDENCIES}"
	"${LIBRARY_EXT_INCLUDES}"
	"${LIBRARY_EXT_LINK_LIBS}"
)

