#include "thorin/pass/copy_prop.h"
#include "thorin/pass/grad_gen.h"
#include "thorin/pass/partial_eval.h"
#include "thorin/pass/reduction.h"
#include "thorin/pass/ret_wrap.h"
#include "thorin/pass/ssa_constr.h"

#include "thorin/transform/compile_ptrns.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

void optimize(World& world) {
#if 1
    PassMan(world)
    .create<PartialEval>()
    .create<Reduction>()
    .create<SSAConstr>()
    .create<CopyProp>()
    .run();
#else
    PassMan(world)
    .create<PartialEval>()
    .create<Reduction>()
    .run();

    PassMan(world)
    .create<SSAConstr>()
    .create<CopyProp>()
    .run();
#endif
}

void optimize_old(World& world) {
    optimize(world);
#if 1
    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);
#endif
    PassMan(world).create<RetWrap>().run();
}

}
