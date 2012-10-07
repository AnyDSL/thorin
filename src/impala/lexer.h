#ifndef IMPALA_LEXER_H
#define IMPALA_LEXER_H

#include <istream>

#include "anydsl/util/location.h"
#include "impala/token.h"

namespace impala {

class Lexer {
public:

    Lexer(std::istream& stream, const std::string& filename);

    /// Get next \p Token in stream.
    Token lex();

private:

    std::ostream& error(const anydsl2::Location& loc);

    /*
     * peek, next and accept
     */

    int next();
    int peek() const { return stream_.peek(); }
    bool accept(int c) {
        if (peek() == c) {
            next();
            return true;
        }
        return false;
    }

    bool accept(std::string& str, int c) {
        if (peek() == c) {
            str += next();
            return true;
        }
        return false;
    }

    bool accept(char c) { return accept((int) c); }
    bool accept(std::string& str, char c) { return accept(str, (int) c); }

    template<class Pred>
    bool accept(std::string& str, Pred pred) {
        if ( pred(peek()) ) {
            str += next();
            return true;
        }
        return false;
    }

    template<class Pred>
    bool accept(Pred pred) {
        if ( pred(peek()) ) {
            next();
            return true;
        }
        return false;
    }

    std::istream& stream_;
    anydsl2::Position pos_;
    anydsl2::Location loc_;
    bool result_;
};

} // namespace impala

#endif
