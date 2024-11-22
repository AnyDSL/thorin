#ifndef THORIN_RUNTIME_H
#define THORIN_RUNTIME_H

/// Backend-agnostic information to interface with the runtime component
namespace thorin {

enum Platform {
    CPU_PLATFORM,
    CUDA_PLATFORM,
    OPENCL_PLATFORM,
    HSA_PLATFORM,
    PAL_PLATFORM,
    LEVEL_ZERO_PLATFORM,
    SHADY_PLATFORM,
};

enum KernelLaunchArgs {
    Mem = 0,
    Device,
    Space,
    Config,
    Body,
    Return,
    Num
};

}

#endif
