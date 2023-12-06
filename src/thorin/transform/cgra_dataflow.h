#ifndef THORIN_TRANSFORM_CGRA_DATAFLOW_H
#define THORIN_TRANSFORM_CGRA_DATAFLOW_H

#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/importer.h"

namespace thorin {

//using Cont2ParamModes     = ContinuationMap<Array<ParamMode>>;
using ContName2ParamModes   = std::vector<std::pair<std::string, Array<ParamMode>>>;
//using CgraDeviceDefs      = std::tuple<PortIndices,ContinuationMap<Array<size_t>>>;
// If it is for each kernel, then a pair is enough otherwise a map is needed
using CgraDeviceDefs        = std::tuple<PortIndices,ContName2ParamModes>;
//using CgraDeviceDefs      = std::tuple<PortIndices,std::pair<Continuation*, Array<ParamMode>>>;

class World;

//PortIndices cgra_dataflow(Importer&, World&, Def2DependentBlocks&);
CgraDeviceDefs cgra_dataflow(Importer&, World&, Def2DependentBlocks&);
void annotate_cgra_graph_modes(Continuation*, const Ports&, Cont2Config&);
void annotate_channel_modes(const Continuation* imported, const ContName2ParamModes, CGRAKernelConfig::Param2Mode&);
void annotate_interface(Continuation* imported, const Continuation* use);

}

#endif
