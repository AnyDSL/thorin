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

        /// \param[in] lam The enclosing λ.
        /// \param[in] def The definition to create the pullback for.
        /// \returns a tuple containing the value equal to the old definition and the pullback.
        ///
        /// See https://arxiv.org/pdf/1810.07951.pdf
        ///
        /// This will recursivly traverse all uses of the definition,
        /// replacing the definition with a J-wrapped definition.
        ///
        /// Example:
        ///
        ///     // f(a, b) = a / (a + b²)
        ///     fn f(a, b, ret) {
        ///         let y₁ = b * b;
        ///         let y₂ = a + y₁;
        ///         let y₃ = a / y₂;
        ///         ret(y₃)
        ///     }
        ///
        /// Goal: Emit pullbacks for all parameters.
        ///
        /// Step 1: emit_pullback(f, a)
        /// - Step 1.1: emit_pullback for all Uses of a
        ///   - Step 1.1.1: emit_pullback(f, a) for Use y₂ = a + y₁
        ///     rewrite `let y₂` -> `let [y₂, B₂] = [a+y₁, λ∂y₂.[∂y₂,∂y₂]]`
        ///     Recurse for y₂
        ///     - Step 1.1.1.1: emit_pullback(f, y₂) for Use y₃ = a / y₂
        ///       rewrite `let y₃` -> `let [y₃, B₃] = [a/y₂, λ∂y₃.[∂y₃/y₂,(-∂y₃*a)/(y₂²)]]`
        ///       Recurse for y₃
        ///       - Step 1.1.1.1.1: emit_pullback(f, y₃) for Use ret(y₃)
        ///         Return reached, end recursion.
        ///         rewrite `ret(y₃)` -> `let [_, ∂y₃] = [[], y₃]`
        ///       emit `let [∂a₂,∂y₂] = B₃(∂y₃)`
        ///     emit `let [∂a₁∂y₁], = B₂(∂y₂)`
        ///   - Step 1.1.2: emit_pullback(f, a) for Use y₃ = a / y₂
        ///     rewrite `let y₃` -> already done
        ///     Recurse for y₃ -> already done
        ///     emit `let [∂a₂,∂y₂] = B₃(∂y₃)` -> equal to prev definition
        /// - Step 1.2: sum pullbacks of all uses
        ///   emit `let ∂a = ∂a₁+∂a₂`
        /// Step 2: emit_pullback(f, b)
        /// - Step 2.1: emit_pullback for all Uses of b
        ///   - Step 2.1.1: emit_pullback(f, b) for Use y₁ = b * b
        ///     rewrite `let y₁`  -> `let [y₁, B₁] = [b*b, λ∂y₁.[b*∂y₁,b*∂y₁]]`
        ///     Recurse for y₁
        ///     - Step 2.1.1.1 emit_pullback(f, a) for Use y₂ = a + y₁
        ///       rewrite `let y₂` -> already done
        ///       Recurse for y₂ -> already done
        ///     emit `let [∂b₁, ∂b₂] = B₁(∂y₁)`
        /// - Step 2.2: sum pullbacks of all uses
        ///   emit `let ∂b = ∂b₁+∂b₂`
        /// Step 3: emit_return
        /// emit `let grads = [∂a,∂b]`
        /// emit `ret(grads)`
        ///
        /// Result:
        ///
        ///    let ∇f = λ[a b ret]. {
        ///        let [y₁, B₁] = [b*b,  λ∂y₁.[b*∂y₁,b*∂y₁]]
        ///        let [y₂, B₂] = [a+y₁, λ∂y₂.[∂y₂,∂y₂]]
        ///        let [y₃, B₃] = [a/y₂, λ∂y₃.[∂y₃/y₂,(-∂y₃*a)/(y₂²)]]
        ///        let [_, ∂y₃] = [[], y₃]
        ///        let [∂a₂,∂y₂] = B₃(∂y₃)
        ///        let [∂a₁,∂y₁] = B₂(∂y₂)
        ///        let ∂a = ∂a₁+∂a₂
        ///        let [∂b₁, ∂b₂] = B₁(∂y₁)
        ///        let ∂b = ∂b₁+∂b₂
        ///        let grads = [∂a,∂b]
        ///        ret(grads)
        ///    }
        const Def* emit_pullback(Lam* lam, const Def* def);

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
