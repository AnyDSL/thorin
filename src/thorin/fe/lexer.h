#ifndef THORIN_FE_LEXER_H
#define THORIN_FE_LEXER_H

#include "thorin/debug.h"
#include "thorin/fe/tok.h"
#include "thorin/util/utf8.h"

namespace thorin {

class World;

class Lexer {
public:
    Lexer(World&, const char*, std::istream&);

    World& world() { return world_; }
    Loc loc() const { return loc_; }
    Tok lex();                                          ///< Get next @p Tok in stream.

private:
    Tok tok(Tok::Tag tag) { return {loc(), tag}; }      ///< Factory method to create a @p Tok.
    bool eof() const { peek(); return stream_.eof(); }  ///< Have we reached the end of file?

    /// @return @c true if @p pred holds.
    /// In this case invoke @p next() and append to @p str_;
    template<class Pred>
    bool accept_if(Pred pred) {
        if (pred(peek())) {
            str_ += next();
            return true;
        }
        return false;
    }

    bool accept(int val) {
        return accept_if([val] (int p) { return p == val; });
    }

    /// Get next byte in @p stream_ and increase @p loc_ / @p peek_pos_.
    int next();
    int peek() const { return stream_.peek(); }
    void eat_comments();

    World& world_;
    Loc loc_;       ///< @p Loc%ation of the @p Tok%en we are currently constructing within @p str_,
    Pos peek_pos_;  ///< @p Pos%ition of the current @p peek().
    std::istream& stream_;
    std::string str_;
    std::unordered_map<std::string, Tok::Tag> keywords_;
};

}

#endif
