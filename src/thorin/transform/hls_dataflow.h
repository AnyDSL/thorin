#ifndef THORIN_TRANSFORM_HLS_DATAFLOW_H
#define THORIN_TRANSFORM_HLS_DATAFLOW_H

#include "thorin/be/kernel_config.h"
#include "thorin/transform/importer.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

enum class ChannelMode : uint8_t {
    Read,       ///< Read-channel
    Write       ///< Write-channe
};
using Def2Mode = DefMap<ChannelMode>;

using Top2Kernel          = std::vector<std::tuple<size_t, std::string, size_t>>;
using HlsDeviceParams     = std::vector<const Def*>;
using Def2DependentBlocks = DefMap<std::pair<Continuation*, Continuation*>>; // [global_def, (HLS_basicblock, CGRA_basicblock)]
using DeviceDefs          = std::tuple<HlsDeviceParams, Def2DependentBlocks>;
class World;

/**
 * implements channels and interface parameters for kernels
 * creates a new kernel named hls_top
 * generates channels in hls_top
 * calls all kernels within hls_top
 * resolves all dependency requirements between kernel calls
 * provides hls_top parameters for hls runtime
 */
//DeviceParams hls_dataflow(Importer&, Top2Kernel&, World&);
//DeviceParams hls_dataflow(Importer&, Top2Kernel&, World&, Importer&);
DeviceDefs hls_dataflow(Importer&, Top2Kernel&, World&, Importer&);
void hls_annotate_top(World&, const Top2Kernel&, Cont2Config&);
void extract_kernel_channels(const Schedule&, Def2Mode&); 
bool is_single_kernel(Continuation*);

}

#endif
