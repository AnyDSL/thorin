#ifndef THORIN_TRANSFORM_COGEN_H
#define THORIN_TRANSFORM_COGEN_H

#include <iostream>
#include "thorin/def.h"
#include "thorin/analyses/bta.h"
#include "thorin/type.h"
#include "thorin/lambda.h"

namespace thorin {

struct CoGen {
    void run(World &world);
    void emit_generator(std::ostream &out, Lambda *lambda);

    private:
    BTA bta;

    void emit_head(std::ostream &out);

    /* Emit staged code. */
    void create_lambda();
    void create_select();
    void create_jump();
    void create_add();
    void create_mul();
};

}

#endif
