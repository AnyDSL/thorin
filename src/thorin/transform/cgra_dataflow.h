#ifndef THORIN_TRANSFORM_CGRA_DATAFLOW_H
#define THORIN_TRANSFORM_CGRA_DATAFLOW_H

#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/importer.h"

namespace thorin {

class World;

//PortIndices cgra_dataflow(Importer&, World&, Def2DependentBlocks&);
CgraDeviceDefs cgra_dataflow(Importer&, World&, Def2DependentBlocks&);

}

#endif
