#include "thorin/fe/parser.h"

namespace thorin {

Sym Parser::parse_sym(const char* ctxt) {
    if (ahead().isa(Tok::Tag::M_id)) return lex().sym();
    err("identifier", ctxt);
    return symtab.make("<error>");
}

Parser::Parser(const char* file, std::istream& stream)
    : lexer_(file, stream)
    , prev_(lexer_.loc())
    , ahead_(lexer_.lex())
{}

Tok Parser::lex() {
    auto result = ahead();
    ahead_ = lexer_.lex();
    return result;
}

bool Parser::accept(Tok::Tag tag) {
    if (tag != ahead().tag()) return false;
    lex();
    return true;
}

bool Parser::expect(Tok::Tag tag, const char* ctxt) {
    if (ahead().tag() == tag) {
        lex();
        return true;
    }

    err(std::string("'") + Tok::tag2str(tag) + std::string("'"), ctxt);
    return false;
}

void Parser::err(const std::string& what, const Tok& tok, const char* ctxt) {
    tok.loc().err() << "expected " << what << ", got '" << tok << "' while parsing " << ctxt << std::endl;
}

Sym Parser::parse_sym(const char* ctxt) {
    if (ahead().isa(Tok::Tag::M_id)) return lex().sym();
    err("identifier", ctxt);
    return symtab.make("<error>");
}

}
