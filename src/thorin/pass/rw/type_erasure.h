#ifndef THORIN_PASS_TYPE_ERASURE_H
#define THORIN_PASS_TYPE_ERASURE_H

#include "thorin/pass/pass.h"

namespace thorin {

class TypeErasure : public RWPass {
public:
    TypeErasure(PassMan& man)
        : RWPass(man, "type_erasure")
    {}

private:
    const Def* rewrite(Def*, const Def*) override;
    const Sigma* convert(const Join*);

};

}

#endif

