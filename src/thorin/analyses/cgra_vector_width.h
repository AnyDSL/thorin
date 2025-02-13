#ifndef THORIN_ANALYSES_CGRA_VECTOR_WIDTH_H
#define THORIN_ANALYSES_CGRA_VECTOR_WIDTH_H

#include "thorin/world.h"
#include "thorin/continuation.h"

namespace thorin {

template<bool allow_scaling>
class CGRAVectorWidthAnalysis {
private:
    World& world_;
    size_t base_size;
    DefMap<size_t> def_2_size;

public:
    CGRAVectorWidthAnalysis(World& world, size_t base_size) : world_(world), base_size(base_size) {}

    World& world() const { return world_; };

    size_t get_width_for(const Continuation* cont);

    size_t get_width_for(const Def* def);
    void register_width(const Def* def, size_t width);
};

}

#endif
