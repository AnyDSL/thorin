#include "thorin/pass/copy_prop.h"
#include "thorin/pass/inliner.h"
#include "thorin/pass/mem2reg.h"
#include "thorin/pass/partial_eval.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/flatten_tuples.h"
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
    optimize(world);
    cleanup_world(world);
    while (partial_evaluation(world, true)); // lower2cff
    flatten_tuples(world);
    cleanup_world(world);
    codegen_prepare(world);
}

}
