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
        return create_ptr(*value.builder_, value.builder_->world().lea(value.def_, offset, offset->debug()));
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

const Def* Value::load(Debug dbg) const {
    switch (kind()) {
        case ImmutableValRef: return def_;
        case MutableValRef:   return builder_->cur_bb->get_value(handle_,type_, name_);
        case PtrRef:          return builder_->load(def_, dbg);
        case AggRef:          return builder_->extract(value_->load(dbg), def_, dbg);
        default: THORIN_UNREACHABLE;
    }
}

void Value::store(const Def* val, Debug dbg) const {
    switch (kind()) {
        case MutableValRef: builder_->cur_bb->set_value(handle_, val); return;
        case PtrRef:        builder_->store(def_, val, dbg); return;
        case AggRef:        value_->store(world().insert(value_->load(dbg), def_, val, dbg), dbg); return;
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
    auto bb = world().basicblock(debug());
    continuation_->jump(bb, {}, debug());
    first_ = false;
    return continuation_ = bb;
}

void Continuation::jump(JumpTarget& jt, Debug dbg) {
    if (!jt.continuation_) {
        jt.continuation_ = this;
        jt.first_ = true;
    } else
        this->jump(jt.untangle(), {}, dbg);
}

Continuation* JumpTarget::branch_to(World& world, Debug dbg) {
    auto bb = world.basicblock(dbg + (continuation_ ? name() + std::string("_crit") : name()));
    bb->jump(*this, dbg);
    bb->seal();
    return bb;
}

Continuation* JumpTarget::enter() {
    if (continuation_ && !first_)
        continuation_->seal();
    return continuation_;
}

Continuation* JumpTarget::enter_unsealed(World& world) {
    return continuation_ ? untangle() : continuation_ = world.basicblock(debug());
}

//------------------------------------------------------------------------------

Continuation* IRBuilder::continuation(Debug dbg) {
    return continuation(world().fn_type(), CC::C, Intrinsic::None, dbg);
}

Continuation* IRBuilder::continuation(const FnType* fn, CC cc, Intrinsic intrinsic, Debug dbg) {
    auto l = world().continuation(fn, cc, intrinsic, dbg);
    if (fn->num_ops() >= 1 && fn->ops().front()->isa<MemType>()) {
        auto param = l->params().front();
        l->set_mem(param);
        if (param->debug().name().empty())
            param->debug().set("mem");
    }

    return l;
}

void IRBuilder::jump(JumpTarget& jt, Debug dbg) {
    if (is_reachable()) {
        cur_bb->jump(jt, dbg);
        set_unreachable();
    }
}

void IRBuilder::branch(const Def* cond, JumpTarget& t, JumpTarget& f, Debug dbg) {
    if (is_reachable()) {
        if (auto lit = cond->isa<PrimLit>()) {
            jump(lit->value().get_bool() ? t : f, dbg);
        } else if (&t == &f) {
            jump(t, dbg);
        } else {
            auto tl = t.branch_to(world_, dbg);
            auto fl = f.branch_to(world_, dbg);
            cur_bb->branch(cond, tl, fl, dbg);
            set_unreachable();
        }
    }
}

const Def* IRBuilder::call(const Def* to, Defs args, const Type* ret_type, Debug dbg) {
    if (is_reachable()) {
        auto p = cur_bb->call(to, args, ret_type, dbg);
        cur_bb = p.first;
        return p.second;
    }
    return nullptr;
}

const Def* IRBuilder::get_mem() { return cur_bb->get_mem(); }
void IRBuilder::set_mem(const Def* mem) { if (is_reachable()) cur_bb->set_mem(mem); }

const Def* IRBuilder::create_frame(Debug dbg) {
    auto enter = world().enter(get_mem(), dbg);
    set_mem(world().extract(enter, 0, dbg));
    return world().extract(enter, 1, dbg);
}

const Def* IRBuilder::alloc(const Type* type, const Def* extra, Debug dbg) {
    if (!extra)
        extra = world().literal_qu64(0, dbg);
    auto alloc = world().alloc(type, get_mem(), extra, dbg);
    set_mem(world().extract(alloc, 0, dbg));
    return world().extract(alloc, 1, dbg);
}

const Def* IRBuilder::load(const Def* ptr, Debug dbg) {
    auto load = world().load(get_mem(), ptr, dbg);
    set_mem(world().extract(load, 0, dbg));
    return world().extract(load, 1, dbg);
}

void IRBuilder::store(const Def* ptr, const Def* val, Debug dbg) {
    set_mem(world().store(get_mem(), ptr, val, dbg));
}

const Def* IRBuilder::extract(const Def* agg, u32 index, Debug dbg) {
    return extract(agg, world().literal_qu32(index, dbg), dbg);
}

const Def* IRBuilder::extract(const Def* agg, const Def* index, Debug dbg) {
    if (auto ld = Load::is_out_val(agg)) {
        if (use_lea(ld->out_val_type()))
            return load(world().lea(ld->ptr(), index, dbg), dbg);
    }
    return world().extract(agg, index, dbg);
}

//------------------------------------------------------------------------------

}
