#include "thorin/pass/fp/beta_red.h"
#include "thorin/pass/fp/copy_prop.h"
#include "thorin/pass/fp/eta_conv.h"
#include "thorin/pass/fp/scalarize.h"
#include "thorin/pass/fp/ssa_constr.h"
#include "thorin/pass/rw/auto_diff.h"
#include "thorin/pass/rw/bound_elim.h"
#include "thorin/pass/rw/partial_eval.h"
#include "thorin/pass/rw/ret_wrap.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

void optimize(World& world) {
    PassMan(world)
    //.add<PartialEval>()
    .add<EtaConv>()
    .add<BetaRed>()
    .add<SSAConstr>()
    .add<CopyProp>()
    //.add<Scalerize>()
    //.add<AutoDiff>()
    .run();

    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);

    PassMan(world)
    //.add<BoundElim>()
    .add<RetWrap>()
    .run();
}

}
