#ifndef THORIN_OFFLOAD_H
#define THORIN_OFFLOAD_H

#include "codegen.h"
#include "thorin/world.h"

namespace thorin {

enum class KernelArgType : uint8_t { Val = 0, Ptr, Struct };

void lower_offload_intrinsics(World&, DeviceBackends&);

}

#endif //THORIN_OFFLOAD_H
