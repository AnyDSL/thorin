#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "thorin/world.h"
#include "thorin/fe/lexer.h"

using namespace thorin;

TEST(Lexer, Toks) {
    World world;
    std::istringstream is("{ } ( ) [ ] ‹ › « » : , . \\ \\/ λ ∀");
    Lexer lexer(world, "<istringstream>", is);

    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_brace_l));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_brace_r));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_paren_l));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_paren_r));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_bracket_l));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_bracket_r));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_angle_l));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_angle_r));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_quote_l));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::D_quote_r));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::P_colon));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::P_comma));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::P_dot));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::B_lam));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::B_forall));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::B_lam));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::B_forall));
    EXPECT_TRUE(lexer.lex().isa(Tok::Tag::M_eof));
}

TEST(Lexer, Loc) {
    World world;
    std::istringstream is(" test  abc    def if  \nwhile λ foo   ");
    Lexer lexer(world, "<istringstream>", is);
    auto t1 = lexer.lex();
    auto t2 = lexer.lex();
    auto t3 = lexer.lex();
    auto t4 = lexer.lex();
    auto t5 = lexer.lex();
    auto t6 = lexer.lex();
    auto t7 = lexer.lex();
    auto t8 = lexer.lex();
    StringStream s;
    s.fmt("{} {} {} {} {} {} {} {}", t1, t2, t3, t4, t5, t6, t7, t8);
    EXPECT_EQ(s.str(), "test abc def if while λ foo <eof>");
    EXPECT_EQ(t1.loc(), Loc("<istringstream>", {1,  2}, {1,  5}));
    EXPECT_EQ(t2.loc(), Loc("<istringstream>", {1,  8}, {1, 10}));
    EXPECT_EQ(t3.loc(), Loc("<istringstream>", {1, 15}, {1, 17}));
    EXPECT_EQ(t4.loc(), Loc("<istringstream>", {1, 19}, {1, 20}));
    EXPECT_EQ(t5.loc(), Loc("<istringstream>", {2,  1}, {2,  5}));
    EXPECT_EQ(t6.loc(), Loc("<istringstream>", {2,  7}, {2,  7}));
    EXPECT_EQ(t7.loc(), Loc("<istringstream>", {2,  9}, {2, 11}));
    EXPECT_EQ(t8.loc(), Loc("<istringstream>", {2, 14}, {2, 14}));
}

TEST(Lexer, Literals) {
}

TEST(Lexer, Utf8) {
}

TEST(Lexer, Eof) {
    World world;
    std::istringstream is("");

    Lexer lexer(world, "<istringstream>", is);
    for (int i = 0; i < 100; i++)
        EXPECT_TRUE(lexer.lex().isa(Tok::Tag::M_eof));
}
