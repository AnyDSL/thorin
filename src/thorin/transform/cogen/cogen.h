#ifndef THORIN_TRANSFORM_COGEN_H
#define THORIN_TRANSFORM_COGEN_H

#include "thorin/def.h"

namespace thorin {

struct CoGen {
    void run(World &world);

    /* Emit staged code. */
    virtual void emit_lambda()    = 0;
    virtual void emit_select()    = 0;
    virtual void emit_jump()      = 0;
    virtual void emit_add()       = 0;
    virtual void emit_mul()       = 0;
};

}

#endif
