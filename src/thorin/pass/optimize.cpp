#include "thorin/pass/fp/beta_red.h"
#include "thorin/pass/fp/copy_prop.h"
#include "thorin/pass/fp/eta_exp.h"
#include "thorin/pass/fp/eta_red.h"
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
    PassMan man1(world);
    //man1.add<BetaRed>();
    //man1.add<PartialEval>();
    auto er = man1.add<EtaRed>();
    man1.add<EtaExp>(er);
    man1.add<SSAConstr>();
    //man1.add<CopyProp>();
    //man1.add<Scalerize>();
    //man1.add<AutoDiff>();
    man1.run();

    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);

    PassMan man2(world);
    //man2.add<BoundElim>();
    man2.add<RetWrap>();
    man2.run();
}

}
