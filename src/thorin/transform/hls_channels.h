#ifndef THORIN_TRANSFORM_HLS_CHANNELS_H
#define THORIN_TRANSFORM_HLS_CHANNELS_H

#include "thorin/be/kernel_config.h"

namespace thorin {

enum class ChannelMode : uint8_t {
    Read,       ///< Read-channel
    Write       ///< Write-channe
};

using Top2Kernel = std::vector<std::tuple<size_t, std::string, size_t>>;
class World;

/**
 * implements channels and interface parameters for kernels
 * creates a new kernel named hls_top
 * generates channels in hls_top
 * calls all kernels within hls_top
 * resolves all dependency requirements between kernel calls
 */
void hls_channels(World&, Top2Kernel&);
void hls_annotate_top(World&, const Top2Kernel&, Cont2Config&);

}

#endif
