#include "thorin/pass/inliner.h"
#include "thorin/pass/mem2reg.h"

namespace thorin {

PassMgr optimizer(World& world) {
    PassMgr result(world);
    //result.create<Mem2Reg>();
    result.create<Inliner>();
    return result;
}

}
