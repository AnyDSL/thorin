#ifndef THORIN_BACKENDS_H
#define THORIN_BACKENDS_H

#include "thorin/transform/importer.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class CodeGen {
protected:
    CodeGen(World& world, bool debug);
public:
    virtual void emit(std::ostream& stream) = 0;

    /// @name getters
    //@{
    World& world() const { return world_; }
    bool debug() const { return debug_; }
    //@}

private:
    World& world_;
    bool debug_;
};

struct LaunchArgs {
    enum {
        Mem = 0,
        Device,
        Space,
        Config,
        Body,
        Return,
        Num
    };
};

struct Backends {
    Backends(World& world, int opt, bool debug);

    Cont2Config kernel_config;
    std::vector<Continuation*> kernels;

    std::unique_ptr<CodeGen> cpu_cg;

    enum { Cuda, NVVM, OpenCL, AMDGPU, HLS, BackendCount };
    std::array<std::unique_ptr<CodeGen>, BackendCount> device_cgs;
private:
    std::vector<Importer> importers_;
};

}

#endif
