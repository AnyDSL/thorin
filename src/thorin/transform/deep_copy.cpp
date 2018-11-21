#include "thorin/transform/deep_copy.h"
#include "thorin/analyses/scope.h"

namespace thorin {

void deep_copy(World& world) {
    Scope::for_each<false>(world, [&](Scope& scope) {
        if (scope.entry()->intrinsic() != Intrinsic::DeepCopy)
            return;
        // TODO
        scope.entry()->dump();

        scope.entry()->intrinsic() = Intrinsic::None;

        scope.update();
    });
}

}
