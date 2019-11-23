#ifndef THORIN_GRAD_GEN_H
#define THORIN_GRAD_GEN_H

#include <optional>
#include <utility>

#include "thorin/pass/pass.h"

namespace thorin {

    class GradGen : public PassBase {
    public:
        GradGen(PassMan& man, size_t index)
            : PassBase(man, index)
        {}

        const Def* rewrite(const Def*) override;

    private:
        const Def* make_gradients(const Lam*);

        struct GradInfo {
            GradInfo(const Pi* grad_type, const Lam* lam, const Uses& uses)
                : grad_type(grad_type), lam(lam), uses(uses)
            {}

            /// Type of the function that results from generating the gradient
            const Pi* grad_type;
            /// The lamda of which the gradient will be generated
            const Lam* lam;
            /// All the uses of the application of the grad-operator that will be replaced by the generated gradient
            const Uses& uses;
        };

        std::optional<GradInfo> isa_grad(const Def*) const;

    };

}

#endif
