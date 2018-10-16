#include "thorin/irbuilder.h"

#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

Value Value::create_val(IRBuilder& builder, const Def* val) {
    Value result;
    result.tag_     = ImmutableValRef;
    result.builder_ = &builder;
    result.def_     = val;
    return result;
}

Value Value::create_mut(IRBuilder& builder, size_t handle, const Type* type) {
    Value result;
    result.tag_     = MutableValRef;
    result.builder_ = &builder;
    result.handle_  = handle;
    result.type_    = type;
    return result;
}

Value Value::create_ptr(IRBuilder& builder, const Def* ptr) {
    Value result;
    result.tag_     = PtrRef;
    result.builder_ = &builder;
    result.def_     = ptr;
    return result;
}

Value Value::create_agg(Value value, const Def* offset) {
    assert(value.tag() != Empty);
    if (value.use_lea())
        return create_ptr(*value.builder_, value.builder_->world().lea(value.def_, offset, offset->debug()));
    Value result;
    result.tag_    = AggRef;
    result.builder_ = value.builder_;
    result.value_.reset(new Value(value));
    result.def_     = offset;
    return result;
}

bool Value::use_lea() const {
    if (tag() == PtrRef)
        return thorin::use_lea(def()->type()->as<PtrType>()->pointee());
    return false;
}

const Def* Value::load(Debug dbg) const {
    switch (tag()) {
        case ImmutableValRef: return def_;
        case MutableValRef:   return builder_->cur_bb->get_value(handle_, type_, dbg);
        case PtrRef:          return builder_->load(def_, dbg);
        case AggRef:          return builder_->extract(value_->load(dbg), def_, dbg);
        default: THORIN_UNREACHABLE;
    }
}

void Value::store(const Def* val, Debug dbg) const {
    switch (tag()) {
        case MutableValRef: builder_->cur_bb->set_value(handle_, val); return;
        case PtrRef:        builder_->store(def_, val, dbg); return;
        case AggRef:        value_->store(world().insert(value_->load(dbg), def_, val, dbg), dbg); return;
        default: THORIN_UNREACHABLE;
    }
}

World& Value::world() const { return builder_->world(); }

//------------------------------------------------------------------------------

#if THORIN_ENABLE_CHECKS
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
    auto bb = world.basicblock(dbg + (continuation_ ? name() + Symbol("_crit") : name()));
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
    if (auto tuple_type = fn->domain()->isa<TupleType>()) {
        if (tuple_type->num_ops() >= 1 && tuple_type->ops().front()->isa<MemType>()) {
            auto param = l->params().front();
            l->set_mem(param);
            if (param->debug().name().empty())
                param->debug().set("mem");
        }
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
            auto tc = t.branch_to(world_, dbg);
            auto fc = f.branch_to(world_, dbg);
            cur_bb->branch(cond, tc, fc, dbg);
            set_unreachable();
        }
    }
}

void IRBuilder::match(const Def* val, JumpTarget& otherwise, Defs patterns, Array<JumpTarget>& targets, Debug dbg) {
    assert(patterns.size() == targets.size());
    if (is_reachable()) {
        if (patterns.size() == 0) return jump(otherwise, dbg);
        if (auto lit = val->isa<PrimLit>()) {
            for (size_t i = 0; i < patterns.size(); i++) {
                if (patterns[i]->as<PrimLit>() == lit)
                    return jump(targets[i], dbg);
            }
            return jump(otherwise, dbg);
        }
        Array<Continuation*> continuations(patterns.size());
        for (size_t i = 0; i < patterns.size(); i++)
            continuations[i] = targets[i].branch_to(world_, dbg);
        cur_bb->match(val, otherwise.branch_to(world_, dbg), patterns, continuations, dbg);
        set_unreachable();
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
    set_mem(world().extract(enter, 0_s, dbg));
    return world().extract(enter, 1, dbg);
}

const Def* IRBuilder::alloc(const Type* type, const Def* extra, Debug dbg) {
    if (!extra)
        extra = world().literal_qu64(0, dbg);
    auto alloc = world().alloc(type, get_mem(), extra, dbg);
    set_mem(world().extract(alloc, 0_s, dbg));
    return world().extract(alloc, 1, dbg);
}

const Def* IRBuilder::load(const Def* ptr, Debug dbg) {
    auto load = world().load(get_mem(), ptr, dbg);
    set_mem(world().extract(load, 0_s, dbg));
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
