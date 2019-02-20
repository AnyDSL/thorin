#include "thorin/pass/inliner.h"

namespace thorin {

PassMgr optimizer(World& world) {
    PassMgr result(world);
    result.create<Inliner>();
    return result;
}

}
