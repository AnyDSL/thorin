#ifndef THORIN_TRANSFORM_HLS_CHANNELS_H
#define THORIN_TRANSFORM_HLS_CHANNELS_H

#include "thorin/be/kernel_config.h"

namespace thorin {

using Top2Kernel = std::vector<std::tuple<size_t, std::string, size_t>>;
class World;

void hls_channels(World&, Top2Kernel&);
void hls_annotate_top(World&, const Top2Kernel&, Cont2Config&);

}

#endif
