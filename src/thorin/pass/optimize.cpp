#include "thorin/pass/copy_prop.h"
#include "thorin/pass/inliner.h"
#include "thorin/pass/mem2reg.h"
#include "thorin/pass/partial_eval.h"

namespace thorin {

void optimize(World& world) {
    PassMan(world)
    .create<Mem2Reg>()
    .create<CopyProp>()
    .create<PartialEval>()
    .create<Inliner>()
    .run();
}

}
