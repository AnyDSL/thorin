#include "thorin/irbuilder.h"

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

Value Value::create_val(IRBuilder& builder, const Def* val) {
    Value result;
    result.kind_    = ImmutableValRef;
    result.builder_ = &builder;
    result.def_     = val;
    return result;
}

Value Value::create_mut(IRBuilder& builder, size_t handle, const Type* type, const char* name) {
    Value result;
    result.kind_    = MutableValRef;
    result.builder_ = &builder;
    result.handle_  = handle;
    result.type_    = type;
    result.name_    = name;
    return result;
}

Value Value::create_ptr(IRBuilder& builder, const Def* ptr) {
    Value result;
    result.kind_    = PtrRef;
    result.builder_ = &builder;
    result.def_     = ptr;
    return result;
}

Value Value::create_agg(Value value, const Def* offset) {
    assert(value.kind() != Empty);
    if (value.use_lea())
        return create_ptr(*value.builder_, value.builder_->world().lea(value.def_, offset, offset->loc()));
    Value result;
    result.kind_    = AggRef;
    result.builder_ = value.builder_;
    result.value_.reset(new Value(value));
    result.def_     = offset;
    return result;
}

bool Value::use_lea() const {
    if (kind() == PtrRef)
        return thorin::use_lea(def()->type()->as<PtrType>()->referenced_type());
    return false;
}

const Def* Value::load(const Location& loc) const {
    switch (kind()) {
        case ImmutableValRef: return def_;
        case MutableValRef:   return builder_->cur_bb->get_value(handle_,type_, name_);
        case PtrRef:          return builder_->load(def_, loc);
        case AggRef:          return builder_->extract(value_->load(loc), def_, loc);
        default: THORIN_UNREACHABLE;
    }
}

void Value::store(const Def* val, const Location& loc) const {
    switch (kind()) {
        case MutableValRef: builder_->cur_bb->set_value(handle_, val); return;
        case PtrRef:        builder_->store(def_, val, loc); return;
        case AggRef:        value_->store(world().insert(value_->load(loc), def_, val, loc), loc); return;
        default: THORIN_UNREACHABLE;
    }
}

World& Value::world() const { return builder_->world(); }

//------------------------------------------------------------------------------

#ifndef NDEBUG
#else
JumpTarget::~JumpTarget() {
    assert((!continuation_ || first_ || continuation_->is_sealed()) && "JumpTarget not sealed");
}
#endif

Continuation* JumpTarget::untangle() {
    if (!first_)
        return continuation_;
    assert(continuation_);
    auto bb = world().basicblock(loc(), name_);
    continuation_->jump(bb, {}, {}, loc());
    first_ = false;
    return continuation_ = bb;
}

void Continuation::jump(JumpTarget& jt, const Location& loc) {
    if (!jt.continuation_) {
        jt.continuation_ = this;
        jt.first_ = true;
    } else
        this->jump(jt.untangle(), {}, {}, loc);
}

Continuation* JumpTarget::branch_to(World& world, const Location& loc) {
    auto bb = world.basicblock(loc, continuation_ ? name_ + std::string("_crit") : name_);
    bb->jump(*this, loc);
    bb->seal();
    return bb;
}

Continuation* JumpTarget::enter() {
    if (continuation_ && !first_)
        continuation_->seal();
    return continuation_;
}

Continuation* JumpTarget::enter_unsealed(World& world) {
    return continuation_ ? untangle() : continuation_ = world.basicblock(loc(), name_);
}

//------------------------------------------------------------------------------

Continuation* IRBuilder::continuation(const Location& loc, const std::string& name) {
    return continuation(world().fn_type(), loc, CC::C, Intrinsic::None, name);
}

Continuation* IRBuilder::continuation(const FnType* fn, const Location& loc, CC cc, Intrinsic intrinsic, const std::string& name) {
    auto l = world().continuation(fn, loc, cc, intrinsic, name);
    if (fn->num_args() >= 1 && fn->args().front()->isa<MemType>()) {
        auto param = l->params().front();
        l->set_mem(param);
        if (param->name.empty())
            param->name = "mem";
    }

    return l;
}

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

const Def* IRBuilder::call(const Def* to, Types type_args, Defs args, const Type* ret_type, const Location& loc) {
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
    set_mem(world().extract(enter, 0, loc));
    return world().extract(enter, 1, loc);
}

const Def* IRBuilder::alloc(const Type* type, const Def* extra, const Location& loc, const std::string& name) {
    if (!extra)
        extra = world().literal_qu64(0, loc);
    auto alloc = world().alloc(type, get_mem(), extra, loc, name);
    set_mem(world().extract(alloc, 0, loc));
    return world().extract(alloc, 1, loc);
}

const Def* IRBuilder::load(const Def* ptr, const Location& loc, const std::string& name) {
    auto load = world().load(get_mem(), ptr, loc, name);
    set_mem(world().extract(load, 0, loc));
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
        if (use_lea(ld->out_val_type()))
            return load(world().lea(ld->ptr(), index, loc, ld->name), loc);
    }
    return world().extract(agg, index, loc, name);
}

//------------------------------------------------------------------------------

}
