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
    PassMan opt(world);
    opt.add<PartialEval>();
    opt.add<BetaRed>();
    auto er = opt.add<EtaRed>();
    opt.add<EtaExp>(er);
    opt.add<SSAConstr>();
    //opt.add<CopyProp>();
    //opt.add<Scalerize>();
    //opt.add<AutoDiff>();
    opt.run();

    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);

    PassMan codgen_prepare(world);
    //codgen_prepare.add<BoundElim>();
    codgen_prepare.add<RetWrap>();
    codgen_prepare.run();
}

}
