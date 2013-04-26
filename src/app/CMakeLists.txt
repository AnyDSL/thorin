
# Boost

SET(Boost_USE_STATIC_LIBS   ON)
SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE( Boost 1.49.0 REQUIRED )


# LLVM

FIND_PACKAGE(LLVM REQUIRED)

IF( NOT LLVM_FOUND )
  MESSAGE(FATAL_ERROR "LLVM package can't be found. Set CMAKE_PREFIX_PATH variable to LLVM's installation prefix.")
ENDIF( NOT LLVM_FOUND )


SET(EXAMPLE_NAME
	anydsl2-app
)

SET(EXAMPLE_DEPENDENCIES
	anydsl2
	impala
)

SET(EXAMPLE_EXT_INCLUDES
	${Boost_INCLUDE_DIR}
	${LLVM_INCLUDE_DIRS}
)

ADD_DEFINITIONS(${LLVM_DEFINITIONS})

llvm_map_components_to_libraries(REQ_LLVM_LIBRARIES jit native analysis)

LINK_DIRECTORIES(
	${LLVM_LIBRARY_DIRS}
)

SET(EXAMPLE_EXT_LINK_LIBS
	${Boost_LIBRARIES}
	${REQ_LLVM_LIBRARIES}
)

add_project_executable(
	"${EXAMPLE_NAME}"
	"automatic"
	"${EXAMPLE_DEPENDENCIES}"
	"${EXAMPLE_EXT_INCLUDES}"
	"${EXAMPLE_EXT_LINK_LIBS}"
)


# install_resource_dir(
	# "${EXAMPLE_NAME}"
	# "${CMAKE_CURRENT_SOURCE_DIR}/resources"
	# "resources"
# )

