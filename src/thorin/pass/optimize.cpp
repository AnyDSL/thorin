#include "thorin/pass/fp/beta_red.h"
#include "thorin/pass/fp/copy_prop.h"
#include "thorin/pass/fp/dce.h"
#include "thorin/pass/fp/eta_exp.h"
#include "thorin/pass/fp/eta_red.h"
#include "thorin/pass/fp/ssa_constr.h"
#include "thorin/pass/rw/bound_elim.h"
#include "thorin/pass/rw/partial_eval.h"
#include "thorin/pass/rw/ret_wrap.h"
#include "thorin/pass/rw/scalarize.h"
#include "thorin/pass/fp/closure_destruct.h"
#include "thorin/pass/fp/unbox_closures.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/transform/closure_conv.h"
#include "thorin/transform/untype_closures.h"

namespace thorin {

void optimize(World& world) {
    // PassMan opt(world);
    // // opt.add<PartialEval>();
    // opt.add<BetaRed>();
    // auto er = opt.add<EtaRed>();
    // auto ee = opt.add<EtaExp>(er);
    // // opt.add<SSAConstr>(ee);
    // // opt.add<CopyProp>();
    // // opt.add<Scalerize>();
    // // opt.add<AutoDiff>();
    // opt.run();

    // while (partial_evaluation(world, true)); // lower2cff
    // world.debug_stream();
    
    ClosureConv(world).run();
    world.debug_stream();
    
    PassMan conv_closures(world);
    // auto br = conv_closures.add<BetaRed>();
    // auto er = conv_closures.add<EtaRed>();
    // auto ee = conv_closures.add<EtaExp>(er);
    // conv_closures.add<CopyProp>(br, ee);
    // conv_closures.add<ClosureDestruct>(ee);
    conv_closures.add<Scalerize>(nullptr);
    conv_closures.add<UnboxClosure>();
    conv_closures.run();

    // PassMan codgen_prepare(world);
    // codgen_prepare.add<Scalerize>(nullptr);
    // codgen_prepare.add<UnboxClosure>();
    // codgen_prepare.add<RetWrap>();
    // codgen_prepare.run();
    // UntypeClosures(world).run();
    // world.debug_stream();
}

}
