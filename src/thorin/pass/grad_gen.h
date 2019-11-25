#ifndef THORIN_GRAD_GEN_H
#define THORIN_GRAD_GEN_H

#include <optional>
#include <utility>
#include <unordered_map>
#include <functional>
#include <bits/stdc++.h>

#include "thorin/pass/pass.h"

namespace thorin {

////////////////////////////////////////////////////////////////////////////////
// environment
////////////////////////////////////////////////////////////////////////////////

struct DefHash {
    inline std::size_t operator()(const Def* def) const { return def->hash(); }
};

/// \brief State of gradient generation
/// It maps:
/// - Variables to their gradients
/// - Variables to all their partial gradients
class GradGenEnv {
public:
    GradGenEnv(World& world) : world_(world) {}

    /// \returns the full gradient for the given var, or nullptr if none exists yet.
    const Def* get_grad(const Def* var);

    /// Saves the partial gradient for the given variable.
    void add_partial_grad(const Def* var, const  Def* partial_grad);

    /// \returns The sum of all partial gradients.
    /// This Removes all duplicated entries from the partial gradients map and
    /// adds it to the full gradients map.
    const Def* sum_partial_grads(const Def* var);

private:
    World& world_;
    std::unordered_map<const Def*, const Def*, DefHash> def_to_grads_;
    std::unordered_multimap<const Def*, const Def*, DefHash> def_to_partial_grads_;
};

////////////////////////////////////////////////////////////////////////////////
// grad-gen pass
////////////////////////////////////////////////////////////////////////////////

class GradGen : public PassBase {
public:
    GradGen(PassMan& man, size_t index)
        : PassBase(man, index), env_(world())
    {}

    /// Finds all uses of the gradient operator and replaces them by a lambda
    /// that calculates the gradients
    const Def* rewrite(const Def*) override;

    /// Keep old code for now
    const Def* rewrite_old(const Def*) ;

private:
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
    /// Step 1: emit_grad(f, a)
    /// - Step 1.1: emit_grad for all Uses of a
    ///   - Step 1.1.1: emit_grad(f, a) for Use y₂ = a + y₁
    ///     rewrite `let y₂` -> `let [y₂, B₂] = [a+y₁, λ∂y₂.[∂y₂,∂y₂]]`
    ///     Recurse for y₂
    ///     - Step 1.1.1.1: emit_grad(f, y₂) for Use y₃ = a / y₂
    ///       rewrite `let y₃` -> `let [y₃, B₃] = [a/y₂, λ∂y₃.[∂y₃/y₂,(-∂y₃*a)/(y₂²)]]`
    ///       Recurse for y₃
    ///       - Step 1.1.1.1.1: emit_grad(f, y₃) for Use ret(y₃)
    ///         Return reached, end recursion.
    ///         rewrite `ret(y₃)` -> `let ∂y₃ = 1.0`
    ///       emit `let [∂a₂,∂y₂] = B₃(∂y₃)`
    ///     emit `let [∂a₁∂y₁], = B₂(∂y₂)`
    ///   - Step 1.1.2: emit_grad(f, a) for Use y₃ = a / y₂
    ///     rewrite `let y₃` -> already done
    ///     Recurse for y₃ -> already done
    ///     emit `let [∂a₂,∂y₂] = B₃(∂y₃)` -> equal to prev definition
    /// - Step 1.2: sum pullbacks of all uses
    ///   emit `let ∂a = ∂a₁+∂a₂`
    /// Step 2: emit_grad(f, b)
    /// - Step 2.1: emit_grad for all Uses of b
    ///   - Step 2.1.1: emit_grad(f, b) for Use y₁ = b * b
    ///     rewrite `let y₁`  -> `let [y₁, B₁] = [b*b, λ∂y₁.[b*∂y₁,b*∂y₁]]`
    ///     Recurse for y₁
    ///     - Step 2.1.1.1 emit_grad(f, a) for Use y₂ = a + y₁
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
    ///        let ∂y₃ = 1.0
    ///        let [∂a₂,∂y₂] = B₃(∂y₃)
    ///        let [∂a₁,∂y₁] = B₂(∂y₂)
    ///        let ∂a = ∂a₁+∂a₂
    ///        let [∂b₁, ∂b₂] = B₁(∂y₁)
    ///        let ∂b = ∂b₁+∂b₂
    ///        let grads = [∂a,∂b]
    ///        ret(grads)
    ///    }
    const Def* emit_grad(Lam* lam, const Def* def);

    /// \returns the J-Call for the given operator.
    /// This is a tuple where the first element is the original value.
    const Def* emit_J(const App* op);
    /// \returns the pullback function for the given operator.
    /// This is a function returning the partial derivatives of the operands.
    const Def* emit_pullback(const App* op);

    /// Finds all the applications of the ∇-axiom that will be replaced by a lambda that
    /// calculates the gradients.
    ///
    /// Due to the structure of the axiom, we search for constructs matching:
    ///     let grad_f = ds2cps(app(app(∇, Σ), cps2ds(λ)))
    Lam* has_lam_to_rewrite(const Def* def) const;

    /// \returns All uses of the variable that are binary operators.
    std::vector<const Def*> uses_are_ops(const Def* use) const;


    GradGenEnv env_;

    ////////////////////////////////////////////////////////////////////////////////
    // Old stuff
    ////////////////////////////////////////////////////////////////////////////////
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
