
############################################################
# MACRO get_internal_files
#
# INTERNAL: find all files macro
#

MACRO(get_internal_files inTraversalMode outInternalFiles inRelativePath)
	FILE(${inTraversalMode} files
		FOLLOW_SYMLINKS
		"${inRelativePath}*.cpp"
		"${inRelativePath}*.c"
		"${inRelativePath}*.cc"
		"${inRelativePath}*.hpp"
		"${inRelativePath}*.h"
		"${inRelativePath}*.inl"
		"${inRelativePath}*.tli"
		"${inRelativePath}*.vs"
		"${inRelativePath}*.gs"
		"${inRelativePath}*.fs"
		"${inRelativePath}*.vert"
		"${inRelativePath}*.frag"
		"${inRelativePath}*.glsl"
		"${inRelativePath}*.shader"
	)
	SET(${outInternalFiles} ${files})
ENDMACRO(get_internal_files)

# create filters macro
#MACRO(CreateFilters files)
#	STRING(LENGTH ${CMAKE_CURRENT_SOURCE_DIR} srcDirLength)
#	FOREACH(file ${files})
#		#truncate the source dir from the filename to have a relative name
#		STRING(LENGTH ${file} fileLength)
#		MATH(EXPR length ${fileLength}-${srcDirLength})
#		STRING(SUBSTRING ${file} ${srcDirLength} ${length} truncatedFile)
#		STRING(REGEX MATCH "^.*/" dirName "${truncatedFile}")
#		STRING(REPLACE "/" "\\" filterName "${dirName}")
#		SOURCE_GROUP(${filterName} FILES ${file})
#	ENDFOREACH(file ${files})
#ENDMACRO(CreateFilters files)

MACRO(postbuild_copy target source dest)
	
	SET(outFile ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${dest})
	
	GET_FILENAME_COMPONENT(destPath ${outFile} PATH)
	#MESSAGE(STATUS "destPath: ${destPath}  --  ${source}  --  ${dest}")
	
	IF(NOT EXISTS ${destPath})
		MESSAGE(STATUS "Create Directory ${destPath}")
		FILE(MAKE_DIRECTORY ${destPath})
	ENDIF(NOT EXISTS ${destPath})
	
	ADD_CUSTOM_COMMAND(TARGET ${target}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${outFile}
		COMMENT "${target}: copying ${source} to ${dest} ..."
	)
	# TODO: specific for each configuration
ENDMACRO(postbuild_copy)

MACRO(install_resource_dir target resDir destDir)

	#postbuild_copy(${target} copy_directory ${resDir} ${destDir})
	
	FILE(GLOB_RECURSE resFiles RELATIVE ${resDir} "${resDir}/*.*")
	#MESSAGE(STATUS "resFiles: ${resFiles}")
	
	FOREACH(rf ${resFiles})
	
		SET(inResFile ${resDir}/${rf})
		SET(outResFile ${destDir}/${rf})
		
		postbuild_copy(${target} ${inResFile} ${outResFile})
		
		GET_FILENAME_COMPONENT(outDir ${outResFile} PATH)
		
		#MESSAGE(STATUS "${inResFile}  --  ${outDir}")
		
		INSTALL(FILES
			${inResFile}
			DESTINATION ${PROJ_BINARY_DIR}/${outDir}
		)	
		
	ENDFOREACH(rf ${resFiles})
	
ENDMACRO(install_resource_dir)

MACRO(install_resource_file target resFile destDir)

	GET_FILENAME_COMPONENT(outFile ${resFile} NAME)
	
	postbuild_copy(${target} ${resFile} "${destDir}/${outFile}")
	
	INSTALL(FILES
		${resFile}
		DESTINATION ${PROJ_BINARY_DIR}/${destDir}
	)
	
ENDMACRO(install_resource_file)

MACRO(install_external_dependency target depFiles)

	FOREACH(depFile ${depFiles})
		#MESSAGE(STATUS "depFile: ${depFile}")
		GET_FILENAME_COMPONENT(outFile ${depFile} NAME)

		postbuild_copy(${target} ${depFile} ${outFile})
	ENDFOREACH()

	INSTALL(FILES
		${depFiles}
		DESTINATION ${PROJ_BINARY_DIR}
		CONFIGURATIONS Release Profiling
	)

ENDMACRO(install_external_dependency)

MACRO(install_external_dependency_debug target depFiles)

	FOREACH(depFile ${depFiles})
		#MESSAGE(STATUS "depFile: ${depFile}")
		GET_FILENAME_COMPONENT(outFile ${depFile} NAME)

		postbuild_copy(${target} ${depFile} ${outFile})
	ENDFOREACH()

	INSTALL(FILES
		${depFiles}
		DESTINATION ${PROJ_BINARY_DIR}
		CONFIGURATIONS Debug
	)

ENDMACRO(install_external_dependency_debug)

MACRO(list_subdirectories retval curdir)
	FILE(GLOB sub-dir RELATIVE ${curdir} *)
	SET(list_of_dirs "")
	FOREACH(dir ${sub-dir})
		IF(IS_DIRECTORY ${curdir}/${dir})
			SET(list_of_dirs ${list_of_dirs} ${dir})
		ENDIF()
	ENDFOREACH()
	SET(${retval} ${list_of_dirs})
ENDMACRO(list_subdirectories)


MACRO(add_subdirectories_with_option)

	SET(optionPrefix ${ARGV0})
	
	SET(curList "None")
	SET(requiredDirectories)
	SET(recommendedDirectories)
	SET(skipDirectories)
	
	FOREACH(arg ${ARGV})
		#MESSAGE(STATUS "ARGV: ${arg}")
		STRING(TOUPPER ${arg} arg_up)
		
		IF(arg_up STREQUAL "REQUIRED")
			SET(curList "REQUIRED")
		ELSEIF(arg_up STREQUAL "RECOMMENDED")
			SET(curList "RECOMMENDED")
		ELSEIF(arg_up STREQUAL "SKIP")
			SET(curList "SKIP")
		ELSE()
			IF(curList STREQUAL "REQUIRED")
				SET(requiredDirectories ${requiredDirectories} ${arg})
			ELSEIF(curList STREQUAL "RECOMMENDED")
				SET(recommendedDirectories ${recommendedDirectories} ${arg})
			ELSEIF(curList STREQUAL "SKIP")
				SET(skipDirectories ${skipDirectories} ${arg})
			ENDIF()
		ENDIF()
	ENDFOREACH()
			
	list_subdirectories(projectDirs ${CMAKE_CURRENT_SOURCE_DIR})
	
	#MESSAGE(STATUS "prefix: ${optionPrefix}")
	#MESSAGE(STATUS "required: ${requiredDirectories}")
	#MESSAGE(STATUS "recommended: ${recommendedDirectories}")
	#MESSAGE(STATUS "skip: ${skipDirectories}")

	FOREACH(proj ${projectDirs})

		SET(proj_cmakefile  ${CMAKE_CURRENT_SOURCE_DIR}/${proj}/CMakeLists.txt)
		
		IF(EXISTS ${proj_cmakefile})

			LIST(FIND skipDirectories ${proj} index)
			
			#MESSAGE(STATUS "${proj} -- ${index}")

			IF(${index} EQUAL -1)

				LIST(FIND requiredDirectories ${proj} index)
				
				#MESSAGE(STATUS "${proj} -- ${index}")

				IF(${index} EQUAL -1)
					SET(required FALSE)
				ELSE(${index} EQUAL -1)
					SET(required TRUE)
				ENDIF(${index} EQUAL -1)
				
				IF(required)
				
					ADD_SUBDIRECTORY(${proj})
					
				ELSE(required)
				
					STRING(TOUPPER ${proj} proj_up)
					SET(OPTIONNAME "${optionPrefix}_${proj_up}")

					LIST(FIND recommendedDirectories ${proj} index)
					
					#MESSAGE(STATUS "${proj} -- ${index}")

					IF(${index} EQUAL -1)
						SET(recommended FALSE)
					ELSE(${index} EQUAL -1)
						SET(recommended TRUE)
					ENDIF(${index} EQUAL -1)
					
					IF(recommended)
						OPTION(${OPTIONNAME} "select to build the ${proj} package of gbpl (recommended)" ON)
					ELSE(recommended)
						OPTION(${OPTIONNAME} "select to build the ${proj} package of gbpl" OFF)
					ENDIF(recommended)
					
					IF(${OPTIONNAME})
						ADD_SUBDIRECTORY(${proj})
					ENDIF(${OPTIONNAME})
					
				ENDIF(required)
				
			ENDIF(${index} EQUAL -1)
			
		ENDIF(EXISTS ${proj_cmakefile})
		
	ENDFOREACH(proj)
ENDMACRO(add_subdirectories_with_option)




########################################################
####
########################################################

############################################################
# MACRO convert_src_to_include_dir
#
# INTERNAL: create include dir from src dir
#

MACRO(convert_src_to_include_dir dir result)
	IF(NOT ${dir} MATCHES "^.*/src(/|$)")
		MESSAGE(WARNING "could not construct include dir for: ${dir}")
	ENDIF()
	STRING(REGEX REPLACE "^(.*/)src(/.*$|$)" "\\1include\\2" ${result} ${dir})
ENDMACRO(convert_src_to_include_dir)


############################################################
# MACRO make_include_dir
#
# INTERNAL: reduce dir to src dir and convert to include
#

MACRO(make_include_dir dir result)
	IF(NOT ${dir} MATCHES "^.*/src(/|$)")
		MESSAGE(WARNING "could not construct include dir for: ${dir}")
	ENDIF()
	STRING(REGEX REPLACE "^(.*/)src(/.*$|$)" "\\1include" ${result} ${dir})
ENDMACRO(make_include_dir)


############################################################
# MACRO make_src_dir
#
# INTERNAL: reduce dir to src dir
#

MACRO(make_src_dir dir result)
	IF(NOT ${dir} MATCHES "^.*/src(/|$)")
		MESSAGE(WARNING "could not construct src dir for: ${dir}")
	ENDIF()
	STRING(REGEX REPLACE "^(.*/)src(/.*$|$)" "\\1src" ${result} ${dir})
ENDMACRO(make_src_dir)


############################################################
# MACRO qt4_auto_wrap
#
# Handles Qt4-specific files (moc, uic, qrc) and adds required include dir
# Call before add_library_cgs/add_executable_cgs!
# outfiles: variable to receive new list of sources
# ...: list of all project sources
#
# example call: qt4_auto_wrap(sources ${sources})
#

MACRO(qt4_auto_wrap outfiles)
    # generate list of moc, uic, qrc files
    FOREACH(fileName ${ARGN})
        # moc: headers (WITH .h) need to contain "Q_OBJECT"
        IF(fileName MATCHES "\\.h$")
            FILE(STRINGS ${fileName} lines REGEX Q_OBJECT)
            IF(lines)
                SET(moc_headers ${moc_headers} ${fileName})
                #MESSAGE(STATUS "moc: ${fileName}")
            ENDIF()
        ENDIF()
        # uic: files have extension ".ui"
        IF(fileName MATCHES "\\.ui$")
            SET(ui_files ${ui_files} ${fileName})
            #MESSAGE(STATUS "uic: ${fileName}")
        ENDIF()
        # qrc: files have extension ".qrc"
        IF(fileName MATCHES "\\.qrc$")
            SET(qrc_files ${qrc_files} ${fileName})
            #MESSAGE(STATUS "qrc: ${fileName}")
        ENDIF()
    ENDFOREACH()
    
    # use standard functions to handle these files
    QT4_WRAP_CPP(${outfiles} ${moc_headers})
    QT4_WRAP_UI(${outfiles} ${ui_files})
    QT4_ADD_RESOURCES(${outfiles} ${qrc_files})
    
    # add include directory for generated ui_*.h files
	INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
ENDMACRO(qt4_auto_wrap)

############################################################
# MACRO extract_headers
#
# Stores all header files (WITH .h) in output
# output: variable to receive list of headers
# ...: list of all project sources
#
# example call: extract_headers(public_headers ${sources})
#

MACRO(extract_headers output)
    SET(headers)
    FOREACH(fileName ${ARGN})
        IF(fileName MATCHES "\\.(h|hpp|inl|tli)$")
            SET(headers ${headers} ${fileName})
        ENDIF()
    ENDFOREACH()
    SET(${output} ${headers})
ENDMACRO()



############################################################
# MACRO install_headers
#
# Adds a post-build step to the target that copies all public header into the include directory
# target: target library
# ...: list of all public headers
#
# example call: install_headers(SOME_LIB ${public_headers})
#

MACRO(install_headers target)
	SET(command)
	# get include dir corresponding to the current source dir
	convert_src_to_include_dir(${CMAKE_CURRENT_SOURCE_DIR} include_dir)
	# create command for copying all files given over there (one by one...)
	FOREACH(fileName ${ARGN})
		FILE(RELATIVE_PATH filePath ${CMAKE_CURRENT_SOURCE_DIR} ${fileName})
		SET(command ${command} COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${filePath}" "${include_dir}/${filePath}")
	ENDFOREACH()

	# add command as post-build step
	ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD ${command} COMMENT "copying public headers...")
ENDMACRO()



############################################################
# MACRO create_filters
#
# Automatically sort sources into filters. Sorting is controlled by two GUI variables (any combination is fine):
# lib_SOURCE_VIEW_SEPARATE_HEADERS_AND_SOURCES: separate headers and sources
# lib_SOURCE_VIEW_FLAT: show files in a flat list or in their subdirectories
# ...: list of all project sources
#
# example call: create_filters(${sources})
#

MACRO(create_filters)
	# go through all files
	FOREACH(filename ${ARGN})
		# if not in flat mode, get path relative to current source dir
		get_filename_component(path "${filename}" REALPATH)
		FILE(RELATIVE_PATH path ${CMAKE_CURRENT_SOURCE_DIR} ${path})
		get_filename_component(path "${path}" PATH)

		STRING(REPLACE "/" "\\" path "${path}")
		
		#MESSAGE(STATUS "filename: ${filename}")

		get_filename_component(name "${filename}" NAME)

		# create separate filter only for generated files
		IF(${name} MATCHES "^inl_|^ui_|cxx$|hxx$|ixx$")
			source_group("_generated" FILES ${filename})
#		ELSEIF(${filename} MATCHES "(h|hpp|cpp|c|inl|tli|cu|ui|qrc)$")
#			source_group("${path}" FILES ${filename})
#		ELSEIF(${filename} MATCHES "(vs|gs|fs|vert|frag|glsl|shader)$")
		ELSE()
			source_group("${path}" FILES ${filename})
		ENDIF()	    
	ENDFOREACH() 
ENDMACRO()


############################################################
# MACRO retrieve_system
#
# saves a "64" for a 64-bit build and a "32" for a 32-bit build

MACRO(retrieve_system suffix)
IF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    SET( ${suffix} "64" )
ELSE( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    SET( ${suffix} "32" )
ENDIF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
ENDMACRO(retrieve_system)



MACRO(add_build_mp_option prefix)
IF(WIN32)
    IF(MSVC)
		SET(option_name "${prefix}_WIN32_USE_MP")
        OPTION(${option_name} "Set to ON to build with the /MP option." OFF)
        MARK_AS_ADVANCED(${option_name})
        IF(${option_name})
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        ENDIF(${option_name})
    ENDIF(MSVC)
ENDIF(WIN32)
ENDMACRO(add_build_mp_option)


MACRO(project_libraries_named proj outVar)
	STRING(TOLOWER ${proj} proj_lo)
	
	SET(${outVar}
		debug ${proj_lo}
		optimized ${proj_lo}
	)
ENDMACRO(project_libraries_named)

MACRO(project_libraries proj)
	STRING(TOUPPER ${proj} proj_up)
	project_libraries_named(${proj} "${proj_up}_LIBRARIES")
ENDMACRO(project_libraries)



#################################################################################
#################################################################################
#################################################################################


MACRO(add_project_part PART_NAME PART_TYPE PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	SET(SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})

	STRING(TOUPPER ${PART_NAME} part_name_up)
	STRING(TOLOWER ${PART_NAME} part_name_lo)
	
	SET(IS_LIB FALSE)
	SET(IS_STATIC FALSE)
	SET(IS_EXE FALSE)
	SET(IS_3RD FALSE)

	IF(${PART_TYPE} STREQUAL "LIB")
		SET(IS_LIB TRUE)
	ELSEIF(${PART_TYPE} STREQUAL "STATIC")
		SET(IS_STATIC TRUE)
	ELSEIF(${PART_TYPE} STREQUAL "EXE")
		SET(IS_EXE TRUE)
	ELSEIF(${PART_TYPE} STREQUAL "3RD")
		SET(IS_3RD TRUE)
	ENDIF()

	
	SET(type_name "Unknown")
	
	IF(IS_LIB OR IS_STATIC)
		SET(type_name "Library")
	ENDIF(IS_LIB OR IS_STATIC)
	IF(IS_EXE)
		SET(type_name "Executable")
	ENDIF(IS_EXE)
	IF(IS_3RD)
		SET(type_name "3rd-party")
	ENDIF(IS_3RD)
	
	MESSAGE(STATUS "Project ${type_name} FOUND: ${part_name_lo}")
	# MESSAGE(STATUS "Project Sources: ${PART_SOURCES}")

	IF("${PART_SOURCES}" STREQUAL "automatic")
		get_internal_files(GLOB_RECURSE internalFiles "")
	ELSE("${PART_SOURCES}" STREQUAL "automatic")
		SET(internalFiles ${PART_SOURCES})
	ENDIF("${PART_SOURCES}" STREQUAL "automatic")
	
	create_filters(${internalFiles})

	INCLUDE_DIRECTORIES(
	#	${SOURCE_DIR}
	#	${GRASS_INCLUDE_DIR}
	#	${THIRDPARTY_DIR}
		${PART_EXT_INCLUDES}
	)

	#LINK_DIRECTORIES(
	#	${GRASS_LIBRARY_DIR}
	#)
	
	IF(IS_LIB)
	
		ADD_DEFINITIONS("-D${part_name_up}_EXPORTS")

		ADD_LIBRARY(${part_name_lo} SHARED
			${internalFiles}
		)
		
		# INSTALL(FILES
			# ${exported_header_files}
			# DESTINATION "include"
		# )
		
	ENDIF(IS_LIB)

	IF(IS_STATIC)
	
		ADD_DEFINITIONS("-D${part_name_up}_STATIC")

		ADD_LIBRARY(${part_name_lo} STATIC
			${internalFiles}
		)
		
		# INSTALL(FILES
			# ${exported_header_files}
			# DESTINATION "include"
		# )
		
	ENDIF(IS_STATIC)
	
	IF(IS_EXE)
		ADD_EXECUTABLE(${part_name_lo}
			${internalFiles}
		)
		
		# actually not necessary - but this breaks somewhere
		SET_TARGET_PROPERTIES(${part_name_lo}
			PROPERTIES
			DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
		)
	ENDIF(IS_EXE)

	IF(IS_3RD)

		ADD_DEFINITIONS("-D${part_name_up}_EXPORTS")

		ADD_LIBRARY(${part_name_lo} SHARED
			${internalFiles}
		)
		
	ENDIF(IS_3RD)
	
	SET(PART_LINK_LIBS)
	
	FOREACH(dep_proj ${PART_DEPENDENCIES})
		project_libraries_named(${dep_proj} libs)
		SET(PART_LINK_LIBS ${PART_LINK_LIBS} ${libs})
	ENDFOREACH()
	
	#MESSAGE(STATUS "Linking:")
	#MESSAGE(STATUS ${PROJECT_LINK_LIBS})
	#MESSAGE(STATUS ${PROJECT_EXT_LINK_LIBS})
	
	TARGET_LINK_LIBRARIES(${part_name_lo}
		${PART_LINK_LIBS}
		${PART_EXT_LINK_LIBS}
	)

	# EnableProfiling(${part_name_lo})
	
	#SET(VERSION_VAR_NAME "${proj_name_lo}_VERSION")
	#SET(${VERSION_VAR_NAME} "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
	#IF(PROJECT_VERSION_BUILD)
	#	SET(${VERSION_VAR_NAME} "${${VERSION_VAR_NAME}}.${PROJECT_VERSION_BUILD}")
	#ENDIF(PROJECT_VERSION_BUILD)
	
	#MESSAGE(STATUS "Version: ${PROJ_VERSION}")
	
	SET_TARGET_PROPERTIES(${part_name_lo}
		PROPERTIES
		PROJECT_LABEL "${type_name} ${part_name_lo}"
		VERSION ${PROJ_VERSION}
		SOVERSION ${PROJ_VERSION}
	)
	
	# INSTALL(TARGETS ${part_name_lo}
		# RUNTIME DESTINATION ${PROJ_BINARY_DIR}
		# LIBRARY DESTINATION ${PROJ_LIBRARY_DIR}
		# ARCHIVE DESTINATION ${PROJ_LIBRARY_DIR}
	# )
	
	IF (IS_LIB OR IS_STATIC)
	
		INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			DESTINATION ${PROJ_INCLUDE_DIR}
			FILES_MATCHING REGEX ".h$|.inl$|.hpp$|.tli$"
		)

	ENDIF (IS_LIB OR IS_STATIC)
	
	
	#MESSAGE(STATUS)
ENDMACRO(add_project_part)

MACRO(add_project_library PART_NAME PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	add_project_part("${PART_NAME}" "LIB" "${PART_SOURCES}" "${PART_DEPENDENCIES}" "${PART_EXT_INCLUDES}" "${PART_EXT_LINK_LIBS}")
ENDMACRO(add_project_library)

MACRO(add_project_static_library PART_NAME PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	add_project_part("${PART_NAME}" "STATIC" "${PART_SOURCES}" "${PART_DEPENDENCIES}" "${PART_EXT_INCLUDES}" "${PART_EXT_LINK_LIBS}")
ENDMACRO(add_project_static_library)

MACRO(add_project_executable PART_NAME PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	add_project_part("${PART_NAME}" "EXE" "${PART_SOURCES}" "${PART_DEPENDENCIES}" "${PART_EXT_INCLUDES}" "${PART_EXT_LINK_LIBS}")
ENDMACRO(add_project_executable)

MACRO(add_project_3rdparty_library PART_NAME PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	add_project_part("${PART_NAME}" "3RD" "${PART_SOURCES}" "${PART_DEPENDENCIES}" "${PART_EXT_INCLUDES}" "${PART_EXT_LINK_LIBS}")
ENDMACRO(add_project_3rdparty_library)

MACRO(add_project_example PART_NAME PART_SOURCES PART_DEPENDENCIES PART_EXT_INCLUDES PART_EXT_LINK_LIBS)
	add_project_part("${PART_NAME}" "EXE" "${PART_SOURCES}" "${PART_DEPENDENCIES}" "${PART_EXT_INCLUDES}" "${PART_EXT_LINK_LIBS}")
ENDMACRO(add_project_example)

