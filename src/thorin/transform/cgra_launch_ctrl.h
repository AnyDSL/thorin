#ifndef THORIN_TRANSFORM_CGRA_LAUNCH_CTRL_H
#define THORIN_TRANSFORM_CGRA_LAUNCH_CTRL_H
#include "thorin/transform/importer.h"

namespace thorin {

class World;

/**
 * removes all calls to individual cgra kernels in host code when launching by FPGA
 * Reactivate the calls when simulating kernels or launching them directly withouth an FPGA
 */
void cgra_launch_ctrl(World&);

}

#endif
