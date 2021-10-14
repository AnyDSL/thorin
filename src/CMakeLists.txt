# build thorin lib
add_subdirectory(thorin)

if(MSVC)
    target_compile_definitions(thorin PRIVATE -D_SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(thorin PRIVATE -D_CRT_SECURE_NO_WARNINGS)
    target_compile_options(thorin PRIVATE "/wd4800" "/wd4520")
    target_compile_options(thorin PRIVATE "/experimental:external" "/external:anglebrackets" "/external:W0")
else()
    target_compile_options(thorin PRIVATE "-Wall" "-Wextra")
endif()

# check for colored terminal
set(COLOR_TTY_AVAILABLE TRUE)
if(WIN32)
    # Windows console does not support ANSI escape codes
    set(COLOR_TTY_AVAILABLE FALSE)
endif()
set(COLORIZE_OUTPUT ${COLOR_TTY_AVAILABLE} CACHE BOOL "Set to true to enable colored error messages")
if(COLORIZE_OUTPUT)
    target_compile_definitions(thorin PRIVATE -DCOLORIZE_LOG)
endif()
