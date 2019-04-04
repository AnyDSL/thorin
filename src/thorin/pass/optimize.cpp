#include "thorin/pass/copy_prop.h"
#include "thorin/pass/inliner.h"
#include "thorin/pass/mem2reg.h"
#include "thorin/pass/partial_eval.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/clone_bodies.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

void optimize(World& world) {
    PassMan(world)
    //.create<CopyProp>()
    .create<PartialEval>()
    .create<Inliner>()
    .create<Mem2Reg>()
    .run();
}

void optimize_old(World& world) {
    cleanup_world(world); // TODO remove this
    optimize(world);
    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    clone_bodies(world);
    lift_builtins(world);
    inliner(world);
    cleanup_world(world);
    codegen_prepare(world);
}

}
