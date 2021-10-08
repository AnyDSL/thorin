#include "thorin/fe/tok.h"

namespace thorin {

const char* Tok::tag2str(Tok::Tag tag) {
    switch (tag) {
#define CODE(t, str) case Tok::Tag::t: return str;
        THORIN_KEY(CODE)
        THORIN_TOK(CODE)
#undef CODE
    }

    return nullptr; // shutup warning
}

Stream& Tok::stream(Stream& s) const {
    if (isa(Tok::Tag::M_id)) return s << sym();
    return s << Tok::tag2str(tag());
}

}
