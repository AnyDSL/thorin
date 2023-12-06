#ifndef THORIN_BE_KERNEL_CONFIG_H
#define THORIN_BE_KERNEL_CONFIG_H

#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/hash.h"
//#include "thorin/continuation.h"

namespace thorin {

// the type of this enum must be .
enum class ChannelMode : int8_t {
    Read = 0,   ///< Read-channel
    Write,      ///< Write-channe
    ReadWrite,  ///< in-out-channel
    Undef,      ///< undefined
    Count
};

typedef ChannelMode ParamMode;

class KernelConfig : public RuntimeCast<KernelConfig> {
public:
    virtual ~KernelConfig() {}
};

typedef ContinuationMap<std::unique_ptr<KernelConfig>> Cont2Config;

class GPUKernelConfig : public KernelConfig {
public:
    GPUKernelConfig(std::tuple<int, int, int> block, bool has_restrict = false)
        : block_(block), has_restrict_(has_restrict)
    {}

    std::tuple<int, int, int> block_size() const { return block_; }

    bool has_restrict() const { return has_restrict_; }

private:
    std::tuple<int, int, int> block_;
    bool has_restrict_;
};

class CGRAKernelConfig : public KernelConfig {
public:
    //TODO: return the modes from cgra_dataflow and pass it to this constructor via param2mode
    using Param2Mode = DefMap<ChannelMode>;
    using Param2Size = GIDMap<const Param*, uint32_t>;
    //CGRAKernelConfig(float runtime_ratio, std::pair<int, int> location, bool has_restrict = false, const Param2Mode& param2mode = {})
    CGRAKernelConfig(float runtime_ratio, std::pair<int, int> location, int vector_size, const Param2Mode& param2mode, bool has_restrict = false)
        : runtime_ratio_(runtime_ratio), location_(location), vector_size_(vector_size), param2mode_(param2mode), has_restrict_(has_restrict)


//    CGRAKernelConfig(float runtime_ratio, std::pair<int, int> location, const Param2Mode& param2mode, const Param2Size& param_sizes, bool has_restrict = false)
//        : runtime_ratio_(runtime_ratio), location_(location), param2mode_(param2mode), param_sizes_(param_sizes), has_restrict_(has_restrict)
    {

       // for (auto param_it = param2mode_.begin(); param_it != param2mode_.end(); ++param_it) {
       //     std::cout << "CTOR param = "; 
       //     auto [param, mode] = *param_it;
       //     param->dump();
       //     if (param_it->second == ChannelMode::Read) std::cout << "Read\n";
       //     else if (param_it->second == ChannelMode::Write) std::cout << "Write\n";
       //     else if (param_it->second == ChannelMode::ReadWrite) std::cout << "ReadWrite\n";
       //     else std::cout << "Undef\n";
       // }


    }

    ChannelMode param_mode(const Param* param) const {
        auto param_it = param2mode_.find(param);
        if (param_it != param2mode_.end()) {
            return param_it->second;
        }
        return ChannelMode::Undef;
    }

    bool has_restrict() const { return has_restrict_; }
    //Interface interface() const { return interface_; }
    float runtime_ratio() const { return runtime_ratio_; }
    std::pair<int, int> location() const { return  location_; }
    int vector_size() const { return vector_size_; }

    uint32_t param_size(const Param* param) const {
        auto it = param_sizes_.find(param);
        if (it != param_sizes_.end())
            return it->second;
        return 0;
    }

private:
    float runtime_ratio_;
    std::pair<int, int> location_;
    int vector_size_;
    bool has_restrict_;
    Param2Mode param2mode_;
    Param2Size param_sizes_;
    //Interface interface_;
};

class HLSKernelConfig : public KernelConfig {
public:
    using Param2Size = GIDMap<const Param*, uint32_t>;
    HLSKernelConfig(const Param2Size& param_sizes)
        : param_sizes_(param_sizes)
    {}

    uint32_t param_size(const Param* param) const {
        auto it = param_sizes_.find(param);
        if (it != param_sizes_.end())
            return it->second;
        return 0;
    }

private:
    Param2Size param_sizes_;
};

}

#endif
