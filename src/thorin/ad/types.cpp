#include <thorin/ad/types.h>
#include <thorin/world.h>

namespace thorin {

const Def* tangent_vector_type(const Def* primal_type) {
    auto& world = primal_type->world();

    if (isa<Tag::Real>(primal_type)) {
        return primal_type;
    }

    if (auto arr = primal_type->isa<Arr>()) {
        auto elem_tangent_type = world.type_tangent_vector(arr->op(1));

        // Array of non-differentiable elements is non-differentiable
        if (auto sigma = elem_tangent_type->isa<Sigma>(); sigma && sigma->num_ops() == 0) {
            return world.sigma();
        }

        return world.arr(arr->op(0), elem_tangent_type);
    }

    if (auto sigma = primal_type->isa<Sigma>()) {
        auto num_ops = sigma->num_ops();

        // Σs with a mem are function parameters.
        if (auto mem = isa<Tag::Mem>(sigma->op(0))) {
            auto params = (num_ops > 2) ? world.sigma(sigma->ops().skip_front()) : sigma->op(1);
            return world.sigma({mem, world.type_tangent_vector(params)});
        }

        Array<const Def*> tangent_vectors(num_ops);
        for (size_t i = 0; i < num_ops; ++i) {
            tangent_vectors[i] = world.type_tangent_vector(sigma->op(i));
        }
        return world.sigma(tangent_vectors);
    }

    // Either non-differentiable or needs inlining.
    return nullptr;
}

}
