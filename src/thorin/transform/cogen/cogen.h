#ifndef THORIN_TRANSFORM_COGEN_H
#define THORIN_TRANSFORM_COGEN_H

#include "thorin/def.h"

namespace thorin {

struct CoGen {
    void run(World &world);


    private:
    /* Emit staged code. */
    void emit_lambda();
    void emit_select();
    void emit_jump();
    void emit_add();
    void emit_mul();
};

}

#endif
