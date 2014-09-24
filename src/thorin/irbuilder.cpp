#include "thorin/irbuilder.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

Var Var::create_val(IRBuilder& builder, Def val) {
    Var result;
    result.kind_    = ImmutableValRef;
    result.builder_ = &builder;
    result.def_     = val;
    return result;
}

Var Var::create_mut(IRBuilder& builder, size_t handle, Type type, const char* name) {
    Var result;
    result.kind_    = MutableValRef;
    result.builder_ = &builder;
    result.handle_  = handle;
    result.type_    = *type;
    result.name_    = name;
    return result;
}

Var Var::create_ptr(IRBuilder& builder, Def ptr) {
    Var result;
    result.kind_    = PtrRef;
    result.builder_ = &builder;
    result.def_     = ptr;
    return result;
}

Var Var::create_agg(Var var, Def offset) {
    assert(var.kind() != Empty);
    if (var.kind() == PtrRef)
        return create_ptr(*var.builder_, var.builder_->world().lea(var.def_, offset));
    Var result;
    result.kind_    = AggRef;
    result.builder_ = var.builder_;
    result.var_.reset(new Var(var));
    result.def_     = offset;
    return result;
}

Def Var::load() const {
    switch (kind()) {
        case ImmutableValRef:   return def_;
        case MutableValRef:     return builder_->cur_bb->get_value(handle_, Type(type_), name_);
        case PtrRef:            return world().load(builder_->get_mem(), def_);
        case AggRef:            return world().extract(var_->load(), def_, "", builder()->get_mem());
        default: THORIN_UNREACHABLE;
    }
}

void Var::store(Def def) const {
    switch (kind()) {
        case MutableValRef: builder_->cur_bb->set_value(handle_, def); return;
        case PtrRef:        builder_->set_mem(world().store(builder_->get_mem(), def_, def)); return;
        case AggRef:        var_->store(world().insert(var_->load(), def_, def)); return;
        default: THORIN_UNREACHABLE;
    }
}

World& Var::world() const { return builder_->world(); }

//------------------------------------------------------------------------------

#ifndef NDEBUG
JumpTarget::~JumpTarget() { assert((!lambda_ || first_ || lambda_->is_sealed()) && "JumpTarget not sealed"); }
#endif

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

Lambda* JumpTarget::untangle() {
    if (!first_)
        return lambda_;
    assert(lambda_);
    auto bb = world().basicblock(name_);
    lambda_->jump(bb, {lambda_->get_mem()});
    first_ = false;
    return lambda_ = bb;
}

void JumpTarget::jump_from(Lambda* bb) {
    if (!lambda_) {
        lambda_ = bb;
        first_ = true;
    } else
        bb->jump(untangle(), {bb->get_mem()});
}

Lambda* JumpTarget::branch_to(World& world) {
    auto bb = world.basicblock(lambda_ ? name_ + std::string(".crit") : name_);
    jump_from(bb);
    bb->seal();
    return bb;
}

Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
    return lambda_ ? untangle() : lambda_ = world.basicblock(name_);
}

//------------------------------------------------------------------------------

void IRBuilder::jump(JumpTarget& jt) {
    if (is_reachable()) {
        jt.jump_from(cur_bb);
        set_unreachable();
    }
}

void IRBuilder::branch(Def cond, JumpTarget& t, JumpTarget& f) {
    if (is_reachable()) {
        if (auto lit = cond->isa<PrimLit>()) {
            jump(lit->value().get_bool() ? t : f);
        } else if (&t == &f) {
            jump(t);
        } else {
            auto tl = t.branch_to(world_);
            auto fl = f.branch_to(world_);
            cur_bb->branch(cond, tl, fl, {get_mem()});
            set_unreachable();
        }
    }
}

Def IRBuilder::call(Def to, ArrayRef<Def> args, Type ret_type) {
    if (is_reachable()) {
        auto p = cur_bb->call(to, args, ret_type);
        cur_bb = p.first;
        return p.second;
    }
    return Def();
}

Def IRBuilder::get_mem() { return cur_bb->get_mem(); }
void IRBuilder::set_mem(Def mem) { if (is_reachable()) cur_bb->set_mem(mem); }

//------------------------------------------------------------------------------

}
