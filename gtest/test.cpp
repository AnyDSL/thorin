#include <gtest/gtest.h>

#include "thorin/world.h"

using namespace thorin;

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
    World w;
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}
