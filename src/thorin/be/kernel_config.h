#ifndef THORIN_BE_KERNEL_CONFIG_H
#define THORIN_BE_KERNEL_CONFIG_H

#include "thorin/util/cast.h"
#include "thorin/util/hash.h"

namespace thorin {

class KernelConfig : public RuntimeCast<KernelConfig> {
public:
    enum class Node { GPUKernelConfig, HLSKernelConfig };

    KernelConfig(Node node)
        : node_(node)
    {}
    virtual ~KernelConfig() {}

    Node node() const { return node_; }

private:
    Node node_;
};

typedef LamMap<std::unique_ptr<KernelConfig>> Cont2Config;

class GPUKernelConfig : public KernelConfig {
public:
    static constexpr auto Node = Node::GPUKernelConfig;

    GPUKernelConfig(std::tuple<int, int, int> block, bool has_restrict = false)
        : KernelConfig(Node)
        , block_(block)
        , has_restrict_(has_restrict)
    {}

    std::tuple<int, int, int> block_size() const { return block_; }

    bool has_restrict() const { return has_restrict_; }

private:
    std::tuple<int, int, int> block_;
    bool has_restrict_;
};

class HLSKernelConfig : public KernelConfig {
public:
    static constexpr auto Node = Node::HLSKernelConfig;
    typedef DefMap<uint32_t> Param2Size;

    HLSKernelConfig(const Param2Size& param_sizes)
        : KernelConfig(Node)
        , param_sizes_(param_sizes)
    {}

    uint32_t param_size(const Def* param) const {
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
