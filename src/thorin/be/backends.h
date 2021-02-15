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

    // TODO use arrays + loops for this
    Importer cuda;
    Importer nvvm;
    Importer opencl;
    Importer amdgpu;
    Importer hls;

    // TODO use arrays + loops for this
    std::unique_ptr<CodeGen> cpu_cg;
    std::unique_ptr<CodeGen> cuda_cg;
    std::unique_ptr<CodeGen> nvvm_cg;
    std::unique_ptr<CodeGen> opencl_cg;
    std::unique_ptr<CodeGen> amdgpu_cg;
    std::unique_ptr<CodeGen> hls_cg;
};

}

#endif
