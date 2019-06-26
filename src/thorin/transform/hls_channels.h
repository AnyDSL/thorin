#ifndef THORIN_TRANSFORM_HLS_CHANNELS_H
#define THORIN_TRANSFORM_HLS_CHANNELS_H

#include "thorin/be/kernel_config.h"

namespace thorin {

using Kernel2Index = std::vector<std::pair<std::string, size_t>>;

class World;

void hls_channels(World&);
void hls_annotate_top(World&, Kernel2Index&);

}

#endif
