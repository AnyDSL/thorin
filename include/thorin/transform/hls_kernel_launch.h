#ifndef THORIN_TRANSFORM_HLS_KERNEL_LAUNCH_H
#define THORIN_TRANSFORM_HLS_KERNEL_LAUNCH_H
#include "thorin/transform/importer.h"
#include "thorin/transform/hls_channels.h"

namespace thorin {

class World;

const auto hls_free_vars_offset = 4; // fn (mem, dev, kernel_ptr, cont, / free_vars /)

/**
 * removes all calls to individual kernels in host code
 * and replaces them with a single call to a kernel named hls_top
 */
void hls_kernel_launch(World&, DeviceParams&);

}

#endif
