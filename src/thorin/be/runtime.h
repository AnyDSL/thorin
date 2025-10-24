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

enum class ParallelForArgs {
    Mem = 0,
    NumThreads,
    Lower,
    Upper,
    Fun,
    Return,
    Num
};

enum class SpawnFibersArgs {
    Mem = 0,
    NumThreads,
    NumBlocks,
    NumWarps,
    Fun,
    Return,
    Num
};

enum class SpawnThreadArgs {
    Mem = 0,
    Fun,
    Return,
    Num
};

enum class SyncArgs {
    Mem = 0,
    Id,
    Return,
};

}

#endif
