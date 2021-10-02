#include "thorin/fe/lexer.h"

#include "thorin/world.h"

namespace thorin {

Lexer::Lexer(World& world, const char* filename, std::istream& stream)
    : world_(world)
    , loc_{filename, {1, 1}, {1, 1}}
    , peek_pos_({1, 1})
    , stream_(stream)
{
    if (!stream_) throw std::runtime_error("stream is bad");
#define CODE(t, str) keywords_[str] = Tok::Tag::t;
    THORIN_KEY(CODE)
#undef CODE
}

int Lexer::next() {
    loc_.finis = peek_pos_;
    int c = stream_.get();

    if (c == '\n') {
        ++peek_pos_.row;
        peek_pos_.col = 1;
    } else {
        ++peek_pos_.col;
    }

    return c;
}

Tok Lexer::lex() {
    while (true) {
        loc_.begin = peek_pos_;
        str_.clear();

        if (eof()) return tok(Tok::Tag::M_eof);
        if (accept_if(isspace)) continue;
        if (accept('=')) return tok(Tok::Tag::P_assign);
        if (accept('.')) return tok(Tok::Tag::P_dot);
        if (accept('(')) return tok(Tok::Tag::D_paren_l);
        if (accept(')')) return tok(Tok::Tag::D_paren_r);
        if (accept('/')) {
            if (accept('*')) {
                eat_comments();
                continue;
            }
            if (accept('/')) {
                while (!eof() && peek() != '\n') next();
                continue;
            }

            //Loc(loc_.file, peek_pos_).err() << "invalid input char '/'; maybe you wanted to start a comment?" << std::endl;
            continue;
        }

        // lex identifier or keyword
        if (accept_if([](int i) { return i == '_' || isalpha(i); })) {
            while (accept_if([](int i) { return i == '_' || isalpha(i) || isdigit(i); })) {}
            if (auto i = keywords_.find(str_); i != keywords_.end()) return tok(i->second); // keyword
            return {loc(), world_.sym(str_, world_.dbg(loc()))};                            // identifier
        }

        //Loc(loc_.file, peek_pos_).err() << "invalid input char: '" << (char) peek() << "'" << std::endl;
        next();
    }
}

void Lexer::eat_comments() {
    while (true) {
        while (!eof() && peek() != '*') next();
        if (eof()) {
            //loc_.err() << "non-terminated multiline comment" << std::endl;
            return;
        }
        next();
        if (accept('/')) break;
    }
}

}
