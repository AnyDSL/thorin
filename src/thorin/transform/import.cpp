#include "thorin/world.h"

namespace thorin {

void import(DefSet& done, World& to, Def def) {
    if (!done.contains(def)) {
    }
    //const World& from = def->world();
}

void import(World& to, Def def) {
    DefSet done;
    import(done, to, def);
}

}
