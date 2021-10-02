#ifndef THOIRN_FE_PARSER_H
#define THOIRN_FE_PARSER_H

#include "thorin/world.h"
#include "thorin/fe/lexer.h"

namespace thorin {

class Parser {
public:
    Parser(const char* file, std::istream& stream);

private:
    Sym parse_sym(const char* ctxt);

    /// Trick to easily keep track of @p Loc%ations.
    class Tracker {
    public:
        Tracker(Parser& parser, const Pos& pos)
            : parser_(parser)
            , pos_(pos)
        {}

        operator Loc() const { return {parser_.prev_.file, pos_, parser_.prev_.finis}; }

    private:
        Parser& parser_;
        Pos pos_;
    };

    /// Factory method to build a @p Tracker.
    Tracker tracker() { return Tracker(*this, ahead().loc().begin); }

    /// Invoke @p Lexer to retrieve next @p Tok%en.
    Tok lex();

    /// Get lookahead.
    Tok ahead() const { return ahead_; }

    /// If @p ahead() is a @p tag, @p lex(), and return @c true.
    bool accept(Tok::Tag tag);

    /// @p lex @p ahead() which must be a @p tag.
    /// Issue @p err%or with @p ctxt otherwise.
    bool expect(Tok::Tag tag, const char* ctxt);

    /// Consume @p ahead which must be a @p tag; @c asserts otherwise.
    Tok eat([[maybe_unused]] Tok::Tag tag) { assert(tag == ahead().tag() && "internal parser error"); return lex(); }

    /// Issue an error message of the form:
    /// <code>expected <what>, got '<tok>' while parsing <ctxt></code>
    void err(const std::string& what, const Tok& tok, const char* ctxt);

    /// Same above but uses @p ahead() as @p tok.
    void err(const std::string& what, const char* ctxt) { err(what, ahead(), ctxt); }

    Lexer lexer_;
    Loc prev_;
    Tok ahead_;
};

}

#endif
