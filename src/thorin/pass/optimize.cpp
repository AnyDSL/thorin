#include "thorin/pass/inliner.h"
#include "thorin/pass/mem2reg.h"

namespace thorin {

void optimize(World& world) {
    PassMgr mgr(world);
    mgr
    .create<Mem2Reg>()
    .create<Inliner>()
    .run();
}

}
