add_executable(thorin-gtest
    lexer.cpp
    test.cpp
)

target_compile_options(thorin-gtest PRIVATE -Wall -Wextra)
target_link_libraries (thorin-gtest gtest_main libthorin)
gtest_discover_tests  (thorin-gtest TEST_PREFIX "thorin.")
