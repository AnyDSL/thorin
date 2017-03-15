INCLUDE_DIRECTORIES ( ${PROJ_SOURCE_DIR} )
LINK_DIRECTORIES ( ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )

IF (MSVC)
    ADD_DEFINITIONS ( -D_SCL_SECURE_NO_WARNINGS )
    ADD_DEFINITIONS ( -D_CRT_SECURE_NO_WARNINGS )
    SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4800 /wd4520" )
    SET ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4800 /wd4520" )
ELSE ()
    ADD_DEFINITIONS ( "-Wall -Wextra" )
ENDIF ()

# check for Half library
FIND_PACKAGE ( Half REQUIRED )
IF ( Half_FOUND )
    MESSAGE ( STATUS "Building with Half library from ${Half_INCLUDE_DIRS}." )
    INCLUDE_DIRECTORIES ( ${Half_INCLUDE_DIRS} )
ENDIF ()

# check for RV
FIND_PACKAGE ( RV QUIET )

IF ( RV_FOUND )
    MESSAGE ( STATUS "Building with RV." )
    ADD_DEFINITIONS ( "-DRV_SUPPORT" )
    INCLUDE_DIRECTORIES ( ${RV_INCLUDE_DIR} )
    LINK_DIRECTORIES ( ${RV_LIBRARY_DIR} )
ELSE ()
    MESSAGE ( STATUS "Building without RV. Specify RV_DIR to compile with RV." )
ENDIF ()

# check for possible llvm extension
FIND_PACKAGE ( LLVM CONFIG QUIET )

IF ( LLVM_FOUND )
    MESSAGE ( STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}" )
    MESSAGE ( STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}" )
    ADD_DEFINITIONS ( ${LLVM_DEFINITIONS} )
    INCLUDE_DIRECTORIES ( ${LLVM_INCLUDE_DIRS} )
ELSE ()
    MESSAGE ( STATUS "Building without LLVM. Specify LLVM_DIR to compile with LLVM." )
ENDIF ()

# check for colored terminal
SET ( COLOR_TTY_AVAILABLE TRUE )
IF ( WIN32 )
    # Windows console does not support ANSI escape codes
    SET ( COLOR_TTY_AVAILABLE FALSE )
ENDIF()
SET ( COLORIZE_OUTPUT ${COLOR_TTY_AVAILABLE} CACHE BOOL "Set to true to enable colored error messages" )
IF ( COLORIZE_OUTPUT )
    ADD_DEFINITIONS ( -DCOLORIZE_LOG )
ENDIF ()

INCLUDE ( thorin-shared )

# build thorin lib
ADD_SUBDIRECTORY ( thorin )
