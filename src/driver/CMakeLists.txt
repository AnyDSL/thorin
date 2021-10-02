add_executable(thorin main.cpp)

target_compile_options(thorin PRIVATE -Wall -Wextra)
target_link_libraries(thorin libthorin)
