#ifndef THORIN_TRANSFORM_HLS_KERNEL_LAUNCH_H
#define THORIN_TRANSFORM_HLS_KERNEL_LAUNCH_H
#include "thorin/transform/importer.h"
#include "thorin/transform/hls_channels.h"

namespace thorin {

class World;

/**
 * removes all calls to individual kernels in host code
 * and replaces them with a single call to a kernel named hls_top
 */
void hls_kernel_launch(World&, DeviceParams&);

}

#endif
