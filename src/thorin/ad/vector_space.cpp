#include <thorin/ad/vector_space.h>
#include <thorin/world.h>

namespace thorin {

const Def* tangent_vector_lit_one(const Def* vector_type) {
    auto& world = vector_type->world();

    if (auto real = isa<Tag::Real>(vector_type)) {
        if(auto width = get_width(real)) {
            return world.lit_real(*width, 1.0);
        }
    }

    if (auto arr = vector_type->isa<Arr>()) {
        // TODO: ???
        (void)arr;
    }

    if (auto sigma = vector_type->isa<Sigma>()) {
        auto num_ops = sigma->num_ops();

        Array<const Def*> ones(num_ops);
        for (size_t i = 0; i < num_ops; ++i) {
            ones[i] = world.lit_tangent_one(sigma->op(i));
        }
        return world.tuple(ones);
    }

    // Not a tangent vector or needs more inlining.
    return nullptr;
}

}

