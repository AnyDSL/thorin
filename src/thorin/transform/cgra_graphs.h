#ifndef THORIN_TRANSFORM_CGRA_GRAPHS_H
#define THORIN_TRANSFORM_CGRA_GRAPHS_H

#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/importer.h"

namespace thorin {

class World;

void cgra_graphs(Importer&, World&, Def2DependentBlocks&);

}

#endif
