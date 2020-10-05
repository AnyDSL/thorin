#include "thorin/pass/copy_prop.h"
#include "thorin/pass/inliner.h"
#include "thorin/pass/ssa_constr.h"
#include "thorin/pass/partial_eval.h"
#include "thorin/pass/grad_gen.h"

#include "thorin/transform/compile_ptrns.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

void optimize(World& world) {
    PassMan(world)
    //.create<PartialEval>()
    .create<Inliner>()
    //.create<CopyProp>()
    .run();

    //PassMan(world)
    //.create<SSAConstr>()
    //.run();

    /*
    PassMan(world)
    .create<CopyProp>()
    .run();
    */
}

void optimize_old(World& world) {
    optimize(world);
    /*
    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);
    codegen_prepare(world);
    */
}

}
