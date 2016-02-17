#include "thorin/irbuilder.h"

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

Var Var::create_val(IRBuilder& builder, const Def* val) {
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

Var Var::create_ptr(IRBuilder& builder, const Def* ptr) {
    Var result;
    result.kind_    = PtrRef;
    result.builder_ = &builder;
    result.def_     = ptr;
    return result;
}

Var Var::create_agg(Var var, const Def* offset) {
    assert(var.kind() != Empty);
    if (var.use_lea())
        return create_ptr(*var.builder_, var.builder_->world().lea(var.def_, offset, offset->loc()));
    Var result;
    result.kind_    = AggRef;
    result.builder_ = var.builder_;
    result.var_.reset(new Var(var));
    result.def_     = offset;
    return result;
}

bool Var::use_lea() const {
    if (kind() == PtrRef)
        return def()->type().as<PtrType>()->referenced_type()->use_lea();
    return false;
}

const Def* Var::load(const Location& loc) const {
    switch (kind()) {
        case ImmutableValRef: return def_;
        case MutableValRef:   return builder_->cur_bb->get_value(handle_, Type(type_), name_);
        case PtrRef:          return builder_->load(def_, loc);
        case AggRef:          return builder_->extract(var_->load(loc), def_, loc);
        default: THORIN_UNREACHABLE;
    }
}

void Var::store(const Def* val, const Location& loc) const {
    switch (kind()) {
        case MutableValRef: builder_->cur_bb->set_value(handle_, val); return;
        case PtrRef:        builder_->store(def_, val, loc); return;
        case AggRef:        var_->store(world().insert(var_->load(loc), def_, val, loc), loc); return;
        default: THORIN_UNREACHABLE;
    }
}

World& Var::world() const { return builder_->world(); }

//------------------------------------------------------------------------------

#ifndef NDEBUG
#else
JumpTarget::~JumpTarget() {
    assert((!lambda_ || first_ || lambda_->is_sealed()) && "JumpTarget not sealed");
}
#endif

Lambda* JumpTarget::untangle() {
    if (!first_)
        return lambda_;
    assert(lambda_);
    auto bb = world().basicblock(loc(), name_);
    lambda_->jump(bb, {}, {}, loc());
    first_ = false;
    return lambda_ = bb;
}

void Lambda::jump(JumpTarget& jt, const Location& loc) {
    if (!jt.lambda_) {
        jt.lambda_ = this;
        jt.first_ = true;
    } else
        this->jump(jt.untangle(), {}, {}, loc);
}

Lambda* JumpTarget::branch_to(World& world, const Location& loc) {
    auto bb = world.basicblock(loc, lambda_ ? name_ + std::string("_crit") : name_);
    bb->jump(*this, loc);
    bb->seal();
    return bb;
}

Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
    return lambda_ ? untangle() : lambda_ = world.basicblock(loc(), name_);
}

//------------------------------------------------------------------------------

void IRBuilder::jump(JumpTarget& jt, const Location& loc) {
    if (is_reachable()) {
        cur_bb->jump(jt, loc);
        set_unreachable();
    }
}

void IRBuilder::branch(const Def* cond, JumpTarget& t, JumpTarget& f, const Location& loc) {
    if (is_reachable()) {
        if (auto lit = cond->isa<PrimLit>()) {
            jump(lit->value().get_bool() ? t : f, loc);
        } else if (&t == &f) {
            jump(t, loc);
        } else {
            auto tl = t.branch_to(world_, loc);
            auto fl = f.branch_to(world_, loc);
            cur_bb->branch(cond, tl, fl, loc);
            set_unreachable();
        }
    }
}

const Def* IRBuilder::call(const Def* to, ArrayRef<Type> type_args, ArrayRef<const Def*> args, Type ret_type, const Location& loc) {
    if (is_reachable()) {
        auto p = cur_bb->call(to, type_args, args, ret_type, loc);
        cur_bb = p.first;
        return p.second;
    }
    return nullptr;
}

const Def* IRBuilder::get_mem() { return cur_bb->get_mem(); }
void IRBuilder::set_mem(const Def* mem) { if (is_reachable()) cur_bb->set_mem(mem); }

const Def* IRBuilder::create_frame(const Location& loc) {
    auto enter = world().enter(get_mem(), loc);
    set_mem(world().extract(enter, 0u, loc));
    return world().extract(enter, 1, loc);
}

const Def* IRBuilder::alloc(Type type, const Def* extra, const Location& loc, const std::string& name) {
    if (!extra)
        extra = world().literal_qu64(0, loc);
    auto alloc = world().alloc(type, get_mem(), extra, loc, name);
    set_mem(world().extract(alloc, 0u, loc));
    return world().extract(alloc, 1, loc);
}

const Def* IRBuilder::load(const Def* ptr, const Location& loc, const std::string& name) {
    auto load = world().load(get_mem(), ptr, loc, name);
    set_mem(world().extract(load, 0u, loc));
    return world().extract(load, 1, loc);
}

void IRBuilder::store(const Def* ptr, const Def* val, const Location& loc, const std::string& name) {
    set_mem(world().store(get_mem(), ptr, val, loc, name));
}

const Def* IRBuilder::extract(const Def* agg, u32 index, const Location& loc, const std::string& name) {
    return extract(agg, world().literal_qu32(index, loc), loc, name);
}

const Def* IRBuilder::extract(const Def* agg, const Def* index, const Location& loc, const std::string& name) {
    if (auto ld = Load::is_out_val(agg)) {
        if (ld->out_val_type()->use_lea())
            return load(world().lea(ld->ptr(), index, loc, ld->name), loc);
    }
    return world().extract(agg, index, loc, name);
}

//------------------------------------------------------------------------------

}
