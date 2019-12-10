#ifndef THORIN_GRAD_GEN_H
#define THORIN_GRAD_GEN_H

#include <functional>

#include "thorin/rewrite.h"
#include "thorin/pass/pass.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class GradEmitter {
public:
    GradEmitter(Lam* orig_lam, Lam* grad_lam);

    Lam* emit_grad_lam();

private:
    /// @name sanity checks
    /// @{
    void check_initial_sanity() { /* TODO */ }
    /// @}
    /// @name top-level steps
    /// @{
    void fill_grads_for_orig_params();
    void set_grad_lam_body();
    /// @}
    /// @name lower-level generators
    /// @{
    const Def* emit_grad_for_var(const Def* var);
    const Def* emit_partial_grad_for_rop_use(const Def* var, const Def* use);
    const Def* emit_partial_grad_for_ret_use(const Def* var, const Def* use);
    const Def* emit_pullback_for_rop(const Def* op);
    /// @}
    /// @name parameters
    /// @{
    size_t num_params() const { return grad_lam_->domain()->num_ops() - 2; }
    const Def* orig_param(size_t i) const;
    const Def* grad_param(size_t i) const;
    /// @}
    /// @name state-handling
    /// @{
    void add_partial_grad(const Def* var, const Def* part_grad);
    void add_partial_grads_for_rop(const Def* var, const Def* op, const Def* pullback, const Def* op_grad);
    /// @}
    /// @name pullbacks
    /// @{
    const Def* pullback_for_add(const Def* op);
    const Def* pullback_for_sub(const Def* op);
    const Def* pullback_for_mul(const Def* op);
    const Def* pullback_for_div(const Def* op);
    /// @}
    /// @name Scoping
    /// @{
    const Def* into_grad_scope(const Def* def);
    /// @}

    using PullbackGenerator = const std::function<const Def* (const Def*)>;

    Lam *orig_lam_;
    Lam *grad_lam_;
    Scope orig_scope_;
    World& world_;
    Rewriter rewriter_;
    Def2Def var_to_grads_;
    Def2Def use_to_pullbacks_;
    World::Sea visited_;
    std::array<PullbackGenerator, Num<ROp>> pullback_gens_;
};

class GradGen : public PassBase {
public:
    GradGen(PassMan& man, size_t index) : PassBase(man, index) {}
    const Def* rewrite(const Def*) override;

private:
    Lam* has_lam_to_rewrite(const Def* def) const;
};

} // namespace thorin

#endif
