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
    Tok lex();

private:
    Tok tok(Tok::Tag tag) { return {loc(), tag}; }
    bool eof() const { return peek_.char_ == (char32_t) std::istream::traits_type::eof(); }

    /// @return @c true if @p pred holds.
    /// In this case invoke @p next() and append to @p str_;
    template<class Pred>
    bool accept_if(Pred pred) {
        if (pred(peek_.char_)) {
            str_ += next();
            return true;
        }
        return false;
    }

    bool accept(char32_t val) {
        return accept_if([val] (char32_t p) { return p == val; });
    }

    /// Get next utf8-char in @p stream_ and increase @p loc_ / @p peek_.pos_.
    char32_t next();
    void eat_comments();

    World& world_;
    Loc loc_; ///< @p Loc%ation of the @p Tok%en we are currently constructing within @p str_,
    struct {
        char32_t char_;
        Pos pos_;
    } peek_;
    std::istream& stream_;
    std::string str_;
    std::unordered_map<std::string, Tok::Tag> keywords_;
};

}

#endif
