#include "thorin/world.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/memop.h"
#include "thorin/type.h"
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/clone_bodies.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/lower2cff.h"
#include "thorin/transform/mem2reg.h"
#include "thorin/transform/memmap_builtins.h"
#include "thorin/transform/merge_lambdas.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/util/array.h"

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define THORIN_BREAK asm("int3");
#else
#define THORIN_BREAK { int* __p__ = nullptr; *__p__ = 42; }
#endif

#ifndef NDEBUG
#define THORIN_CHECK_BREAK(gid) \
    if (breakpoints_.find((gid)) != breakpoints_.end()) { THORIN_BREAK }
#else
#define THORIN_CHECK_BREAK(gid) {}
#endif

namespace thorin {

/*
 * constructor and destructor
 */

World::World(std::string name)
    : name_(name)
    , gid_(0)
    , tuple0_ (*unify(new TupleTypeNode(*this, ArrayRef<Type>())))
    , fn0_    (*unify(new FnTypeNode   (*this, ArrayRef<Type>())))
    , mem_    (*unify(new MemTypeNode  (*this)))
    , frame_  (*unify(new FrameTypeNode(*this)))
#define THORIN_ALL_TYPE(T) ,T##_(*unify(new PrimTypeNode(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

World::~World() {
    for (auto primop : primops_) delete primop;
    for (auto type   : types_  ) delete type;
    for (auto lambda : lambdas_) delete lambda;
}

Array<Lambda*> World::copy_lambdas() const {
    Array<Lambda*> result(lambdas().size());
    std::copy(lambdas().begin(), lambdas().end(), result.begin());
    return result;
}

std::vector<Lambda*> World::externals() const {
    std::vector<Lambda*> result;
    for (auto lambda : lambdas_) {
        if (lambda->attribute().is(Lambda::Extern))
            result.push_back(lambda);
    }
    return result;
}

/*
 * types
 */

StructType World::struct_type(size_t size, const std::string& name) {
    assert(false && "TODO");
    return StructType();
}

/*
 * literals
 */

Def World::literal(PrimTypeKind kind, int64_t value, size_t length) {
    Def lit;
    switch (kind) {
#define THORIN_I_TYPE(T) case PrimType_##T:  lit = literal(T(value), 1); break;
#define THORIN_F_TYPE(T) THORIN_I_TYPE(T)
#include "thorin/tables/primtypetable.h"
                         case PrimType_bool: lit = literal(bool(value), 1); break;
            default: THORIN_UNREACHABLE;
    }

    return vector(lit, length);
}

Def World::literal(PrimTypeKind kind, Box box, size_t length) { return vector(cse(new PrimLit(*this, kind, box, "")), length); }
Def World::any    (Type type, size_t length) { return vector(cse(new Any(type, "")), length); }
Def World::bottom (Type type, size_t length) { return vector(cse(new Bottom(type, "")), length); }
Def World::zero   (Type type, size_t length) { return zero  (type.as<PrimType>()->primtype_kind(), length); }
Def World::one    (Type type, size_t length) { return one   (type.as<PrimType>()->primtype_kind(), length); }
Def World::allset (Type type, size_t length) { return allset(type.as<PrimType>()->primtype_kind(), length); }

/*
 * create
 */

Def World::binop(int kind, Def cond, Def lhs, Def rhs, const std::string& name) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, cond, lhs, rhs);

    assert(is_cmp(kind) && "must be a Cmp");
    return cmp((CmpKind) kind, cond, lhs, rhs);
}

Def World::arithop(ArithOpKind kind, Def cond, Def a, Def b, const std::string& name) {
    assert(a->type() == b->type());
    assert(a->type().as<PrimType>()->length() == b->type().as<PrimType>()->length());
    PrimTypeKind type = a->type().as<PrimType>()->primtype_kind();

    // bottom op bottom -> bottom
    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type);

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto lvec = a->isa<Vector>();
    auto rvec = b->isa<Vector>();

    if (lvec && rvec) {
        auto cvec = cond->isa<Vector>();
        size_t num = lvec->type().as<PrimType>()->length();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cvec && cvec->op(i)->is_zero() ? bottom(type, 1) :  arithop(kind, lvec->op(i), rvec->op(i));
        return vector(ops, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();

        try {
            switch (kind) {
                case ArithOp_add:
                    switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_sub:
                    switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_mul:
                    switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_div:
                    switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_rem:
                    switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() % r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_and:
                    switch (type) {
#define THORIN_I_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_or:
                    switch (type) {
#define THORIN_I_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_xor:
                    switch (type) {
#define THORIN_I_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shl:
                    switch (type) {
#define THORIN_I_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shr:
                    switch (type) {
#define THORIN_I_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
            }
        } catch (BottomException) {
            return bottom(type);
        }
    }

    if (a == b) {
        switch (kind) {
            case ArithOp_add:
                if (is_type_i(type))
                    return arithop_mul(cond, literal(type, 2), a);
                else
                    break;

            case ArithOp_sub:
            case ArithOp_rem:
            case ArithOp_xor: return zero(type);

            case ArithOp_div: return one(type);

            case ArithOp_and:
            case ArithOp_or:  return a;

            default: break;
        }
    }

    if (a->is_zero()) {
        switch (kind) {
            case ArithOp_div:
            case ArithOp_rem: return bottom(type);

            case ArithOp_shl:
            case ArithOp_shr: return a;

            default: break;
        }
    } else if (a->is_one()) {
        switch (kind) {
            case ArithOp_div: return a;
            case ArithOp_rem: return zero(type);

            default: break;
        }
    } else if (rlit && rlit->primlit_value<uint64_t>() >= uint64_t(num_bits(type))) {
        switch (kind) {
            case ArithOp_shl:
            case ArithOp_shr: return bottom(type);

            default: break;
        }
    }

    if (kind == ArithOp_xor && a->is_allset()) {    // is this a NOT
        if (b->is_not())                            // do we have ~~x?
            return b->as<ArithOp>()->rhs();
        if (auto cmp = b->isa<Cmp>())   // do we have ~(a cmp b)?
            return this->cmp(negate(cmp->cmp_kind()), cond, cmp->lhs(), cmp->rhs());
    }

    auto lcmp = a->isa<Cmp>();
    auto rcmp = b->isa<Cmp>();

    if (kind == ArithOp_or && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
            && lcmp->cmp_kind() == negate(rcmp->cmp_kind()))
            return literal_bool(true);

    if (kind == ArithOp_and && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
            && lcmp->cmp_kind() == negate(rcmp->cmp_kind()))
            return literal_bool(false);

    auto land = a->kind() == Node_and ? a->as<ArithOp>() : nullptr;
    auto rand = b->kind() == Node_and ? b->as<ArithOp>() : nullptr;

    // distributivity (a and b) or (a and c)
    if (kind == ArithOp_or && land && rand) {
        if (land->lhs() == rand->lhs())
            return arithop_and(cond, land->lhs(), arithop_or(cond, land->rhs(), rand->rhs()));
        if (land->rhs() == rand->rhs())
            return arithop_and(cond, land->rhs(), arithop_or(cond, land->lhs(), rand->lhs()));
    }

    auto lor = a->kind() == Node_or ? a->as<ArithOp>() : nullptr;
    auto ror = b->kind() == Node_or ? b->as<ArithOp>() : nullptr;

    // distributivity (a or b) and (a or c)
    if (kind == ArithOp_and && lor && ror) {
        if (lor->lhs() == ror->lhs())
            return arithop_or(cond, lor->lhs(), arithop_and(cond, lor->rhs(), ror->rhs()));
        if (lor->rhs() == ror->rhs())
            return arithop_or(cond, lor->rhs(), arithop_and(cond, lor->lhs(), ror->lhs()));
    }

    // absorption
    if (kind == ArithOp_and) {
        if (ror) {
            if (a == ror->lhs()) return ror->rhs();
            if (a == ror->rhs()) return ror->lhs();
        }
        if (lor) {
            if (a == lor->lhs()) return lor->rhs();
            if (a == lor->rhs()) return lor->lhs();
        }
    }

    // absorption
    if (kind == ArithOp_or) {
        if (rand) {
            if (a == rand->lhs()) return rand->rhs();
            if (a == rand->rhs()) return rand->lhs();
        }
        if (land) {
            if (a == land->lhs()) return land->rhs();
            if (a == land->rhs()) return land->lhs();
        }
    }

    if (kind == ArithOp_or) {
        if (lor && ror) {
            if (lor->lhs() == ror->lhs())
                return arithop_or(lor->rhs(), ror->rhs());
            if (lor->rhs() == ror->rhs())
                return arithop_or(lor->lhs(), ror->lhs());
        }
    }

    if (kind == ArithOp_and) {
        if (land && rand) {
            if (land->lhs() == rand->lhs())
                return arithop_and(land->rhs(), rand->rhs());
            if (land->rhs() == rand->rhs())
                return arithop_and(land->lhs(), rand->lhs());
        }
    }

    // normalize: a - b = a + -b
    if (kind == ArithOp_sub && !a->is_minus_zero()) {
        rlit = (b = arithop_minus(b))->isa<PrimLit>();
        kind = ArithOp_add;
    }

    // normalize: swap literal/vector to the left
    if (is_commutative(kind) && (rlit || rvec)) {
        std::swap(a, b);
        std::swap(llit, rlit);
        std::swap(lvec, rvec);
    }

    if (a->is_zero()) {
        switch (kind) {
            case ArithOp_mul:
            case ArithOp_div:
            case ArithOp_rem:
            case ArithOp_and:
            case ArithOp_shl:
            case ArithOp_shr: return zero(type);

            case ArithOp_add:
            case ArithOp_or:
            case ArithOp_xor:  return b;

            default: break;
        }
    }
    if (a->is_one()) {
        switch (kind) {
            case ArithOp_mul: return b;
            default: break;
        }
    }
    if (a->is_allset()) {
        switch (kind) {
            case ArithOp_and: return b;
            case ArithOp_or:  return llit; // allset
            default: break;
        }
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(kind)) {
        auto a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_kind() == kind ? a->as<ArithOp>() : nullptr;
        auto b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_kind() == kind ? b->as<ArithOp>() : nullptr;
        auto a_lhs_lv = a_same && (a_same->lhs()->isa<PrimLit>() || a_same->lhs()->isa<Vector>()) ? a_same->lhs() : nullptr;
        auto b_lhs_lv = b_same && (b_same->lhs()->isa<PrimLit>() || b_same->lhs()->isa<Vector>()) ? b_same->lhs() : nullptr;

        if (is_commutative(kind)) {
            if (a_lhs_lv && b_lhs_lv)
                return arithop(kind, arithop(kind, a_lhs_lv, b_lhs_lv), arithop(kind, a_same->rhs(), b_same->rhs()));
            if ((llit || lvec) && b_lhs_lv)
                return arithop(kind, arithop(kind, a, b_lhs_lv), b_same->rhs());
            if (b_lhs_lv)
                return arithop(kind, b_lhs_lv, arithop(kind, a, b_same->rhs()));
        }
        if (a_lhs_lv)
            return arithop(kind, a_lhs_lv, arithop(kind, a_same->rhs(), b));
    }

    return cse(new ArithOp(kind, cond, a, b, name));
}

Def World::arithop_not(Def cond, Def def) { return arithop_xor(cond, allset(def->type(), def->length()), def); }

Def World::arithop_minus(Def cond, Def def) {
    switch (PrimTypeKind kind = def->type().as<PrimType>()->primtype_kind()) {
#define THORIN_F_TYPE(T) \
        case PrimType_##T: \
            return arithop_sub(cond, literal_##T(-0.f, def->length()), def);
#include "thorin/tables/primtypetable.h"
        default:
            assert(is_type_i(kind));
            return arithop_sub(cond, zero(kind), def);
    }
}

Def World::cmp(CmpKind kind, Def cond, Def a, Def b, const std::string& name) {
    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type_bool());

    CmpKind oldkind = kind;
    switch (kind) {
        case Cmp_gt:  kind = Cmp_lt; break;
        case Cmp_ge:  kind = Cmp_le; break;
        default: break;
    }

    if (oldkind != kind)
        std::swap(a, b);

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto  lvec = a->isa<Vector>();
    auto  rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type().as<PrimType>()->length();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cmp(kind, lvec->op(i), rvec->op(i));
        return vector(ops, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();
        PrimTypeKind type = llit->primtype_kind();

        // TODO unordered
        switch (kind) {
            case Cmp_eq:
                switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal_bool(l.get_##T() == r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_ne:
                switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal_bool(l.get_##T() != r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_lt:
                switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal_bool(l.get_##T() <  r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_le:
                switch (type) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: return literal_bool(l.get_##T() <= r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            default: THORIN_UNREACHABLE;
        }
    }

    if (a == b) {
        switch (kind) {
            case Cmp_lt:
            case Cmp_eq:  return zero(type_bool());
            case Cmp_le:
            case Cmp_ne:  return one(type_bool());
            default: break;
        }
    }

    return cse(new Cmp(kind, cond, a, b, name));
}

#if 0

static i64 box2i64(PrimTypeKind kind, Box box) {
    switch (kind) {
#define THORIN_U_TYPE(T) case PrimType_##T: return (i64) (make_signed<T>::type) box.get_##T();
#include "thorin/tables/primtypetable.h"
        THORIN_NO_F_TYPE;
        default: THORIN_UNREACHABLE;
    }
}

Def World::convop(ConvOpKind kind, Def cond, Def from, Type to, const std::string& name) {
#define from_kind (from->type().as<PrimType>()->primtype_kind())
#define   to_kind (  to        .as<PrimType>()->primtype_kind())
#ifndef NDEBUG
    switch (kind) {
        case ConvOp_trunc:      assert(num_bits(from_kind) > num_bits(to_kind)); break;
        case ConvOp_sext:
        case ConvOp_zext:       assert(num_bits(from_kind) < num_bits(to_kind)); break;
        case ConvOp_stof:
        case ConvOp_utof:       assert(  is_int(from_kind) && is_float(to_kind)); break;
        case ConvOp_ftos:
        case ConvOp_ftou:       assert(is_float(from_kind) &&   is_int(to_kind)); break;
        case ConvOp_ftrunc:     assert(from_kind == PrimType_f64 && to_kind == PrimType_f32); break;
        case ConvOp_fext:       assert(from_kind == PrimType_f32 && to_kind == PrimType_f64); break;
        case ConvOp_inttoptr:   assert(is_int(from_kind) && to->isa<Ptr>()); break;
        case ConvOp_ptrtoint:   assert(from->type()->isa<Ptr>() && is_int(to_kind)); break;
        case ConvOp_bitcast:    /* TODO check */;
    }
#endif

    if (from->isa<Bottom>())
        return bottom(to);

    auto lit = from->isa<PrimLit>();
    auto vec = from->isa<Vector>();

    if (vec) {
        auto cvec = cond->isa<Vector>();
        size_t num = vec->length();
        Array<Def> ops(num);
        auto to_scalar = to->as<VectorType>()->scalarize();
        for (size_t i = 0; i != num; ++i)
            ops[i] = cvec && cvec->op(i)->is_zero() ? bottom(to_scalar, 1) :  convop(kind, vec->op(i), to_scalar);
        return vector(ops, name);
    }

    if (lit) {
        Box box = lit->value();

        switch (kind) {
            case ConvOp_trunc:
            case ConvOp_zext:   return literal(to_kind, box);
            case ConvOp_sext:   return literal(to_kind, Box((u64) box2i64(from_kind, box)));
            case ConvOp_utof:
                switch (to_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((T) box.get_u64()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_stof:
                switch (to_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((T) box2i64(from_kind, box)));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftou:
                switch (from_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((u64) box.get_##T()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftos:
                switch (from_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((u64) (i64) box.get_##T()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftrunc: return literal(PrimType_f32, Box((f32) box.get_f64()));
            case ConvOp_fext:   return literal(PrimType_f64, Box((f64) box.get_f32()));
            case ConvOp_bitcast:
            case ConvOp_inttoptr:
            case ConvOp_ptrtoint: /* FALLTROUGH */;
        }
    }

    return cse(new ConvOp(kind, cond, from, to, name));
}
#endif

Def World::extract(Def agg, Def index, const std::string& name) {
    if (agg->isa<Bottom>())
        return bottom(Extract::type(agg, index));

    if (auto aggregate = agg->isa<Aggregate>())
        if (auto lit = index->isa<PrimLit>())
            return aggregate->op_via_lit(lit);

    if (auto insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else if (index->isa<PrimLit>()) {
            if (insert->index()->isa<PrimLit>())
                return extract(insert->agg(), index);
        }
    }

    return cse(new Extract(agg, index, name));
}

Def World::insert(Def agg, Def index, Def value, const std::string& name) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto literal = index->isa<PrimLit>()) {
            Array<Def> args(agg->size());
            std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
            args[literal->primlit_value<u64>()] = value;
            return rebuild(aggregate, args);
        }
    }

    return cse(new Insert(agg, index, value, name));
}

Def World::extract(Def tuple, u32 index, const std::string& name) { return extract(tuple, literal_qu32(index), name); }
Def World::insert(Def tuple, u32 index, Def value, const std::string& name) {
    return insert(tuple, literal_qu32(index), value, name);
}

Def World::vector(Def arg, size_t length, const std::string& name) {
    if (length == 1)
        return arg;

    Array<Def> args(length);
    std::fill(args.begin(), args.end(), arg);
    return vector(args, name);
}

Def World::select(Def cond, Def a, Def b, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (auto lit = cond->isa<PrimLit>())
        return lit->value().get_bool() ? a : b;

    if (cond->is_not()) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    if (a == b)
        return a;

    return cse(new Select(cond, a, b, name));
}

const Enter* World::enter(Def mem, const std::string& name) {
    if (auto leave = mem->isa<Leave>())
        return leave->frame();

    return cse(new Enter(mem, name));
}

Def World::leave(Def mem, Def frame, const std::string& name) {
    for (auto use : frame->uses()) {
        if (use->isa<Slot>())
            return cse(new Leave(mem, frame, name));
    }

    return mem;
}

const Map* World::map(Def mem, Def ptr, uint32_t device, AddressSpace addr_space, Def tleft, Def size, const std::string& name) {
    return cse(new Map(mem, ptr, device, addr_space, tleft, size, name));
}

const Map* World::map(Def mem, Def ptr, Def device, Def addr_space, Def tleft, Def size, const std::string& name) {
    return map(mem, ptr, device->as<PrimLit>()->ps32_value().data(), 
            (AddressSpace)addr_space->as<PrimLit>()->ps32_value().data(), tleft, size, name);
}

const Unmap* World::unmap(Def mem, Def ptr, uint32_t device, AddressSpace addr_space, const std::string& name) {
    return cse(new Unmap(mem, ptr, device, addr_space, name));
}

const Unmap* World::unmap(Def mem, Def ptr, Def device, Def addr_space, const std::string& name) {
    return unmap(mem, ptr, device->as<PrimLit>()->ps32_value().data(), 
            (AddressSpace)addr_space->as<PrimLit>()->ps32_value().data(), name);
}

Def World::load(Def mem, Def ptr, const std::string& name) {
    if (auto store = mem->isa<Store>())
        if (store->ptr() == ptr) {
            return store->val();
    }

    if (auto global = ptr->isa<Global>()) {
        if (!global->is_mutable())
            return global->init();
    }


    return cse(new Load(mem, ptr, name));
}

const Store* World::store(Def mem, Def ptr, Def value, const std::string& name) {
    if (auto store = mem->isa<Store>()) {
        if (ptr == store->ptr())
            mem = store->mem();
    }

    return cse(new Store(mem, ptr, value, name));
}

const LEA* World::lea(Def ptr, Def index, const std::string& name) { return cse(new LEA(ptr, index, name)); }
const Global* World::global(Def init, bool is_mutable, const std::string& name) { return cse(new Global(init, is_mutable, name)); }
const Slot* World::slot(Type type, Def frame, size_t index, const std::string& name) {
    return cse(new Slot(type, frame, index, name));
}

const Global* World::global_immutable_string(const std::string& str, const std::string& name) {
    size_t size = str.size() + 1;

    Array<Def> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = literal_qu8(str[i]);
    str_array.back() = literal_qu8('\0');

    return global(array(str_array), false, name);
}

Def World::run(Def def, const std::string& name) {
    if (auto run = def->isa<Run>()) return run;
    if (auto hlt = def->isa<Hlt>()) return hlt;
    return cse(new Run(def, name));
}

Def World::hlt(Def def, const std::string& name) {
    if (auto hlt = def->isa<Hlt>()) return hlt;
    if (auto run = def->isa<Run>()) def = run->def();
    return cse(new Hlt(def, name));
}

Lambda* World::lambda(FnType fn, Lambda::Attribute attribute, const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto l = new Lambda(gid_++, fn, attribute, true, name);
    lambdas_.insert(l);

    size_t i = 0;
    for (auto elem : fn->elems())
        l->params_.push_back(param(elem, l, i++));

    return l;
}

Lambda* World::meta_lambda() {
    auto l = lambda(fn_type(), "meta");
    l->jump(bottom(fn_type()), {});
    return l;
}

Lambda* World::basicblock(const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto bb = new Lambda(gid_++, fn_type(), Lambda::Attribute(0), false, name);
    lambdas_.insert(bb);
    return bb;
}

Def World::rebuild(World& to, const PrimOp* in, ArrayRef<Def> ops, Type type) {
    NodeKind kind = in->kind();
    const std::string& name = in->name;
    assert(&type->world() == &to);
#ifndef NDEBUG
    for (auto op : ops)
        assert(&op->world() == &to);
#endif

    if (is_arithop (kind))  { assert(ops.size() == 3); return to.arithop((ArithOpKind) kind, ops[0], ops[1], ops[2], name); }
    if (is_cmp     (kind))  { assert(ops.size() == 3); return to.cmp(    (CmpKind)     kind, ops[0], ops[1], ops[2], name); }
    // TODO
    //if (is_convop  (kind))  { assert(ops.size() == 2); return to.convop( (ConvOpKind)  kind, ops[0], ops[1],   type, name); }
    if (is_primtype(kind)) {
        assert(ops.size() == 0);
        auto primlit = in->as<PrimLit>();
        return to.literal(primlit->primtype_kind(), primlit->value());
    }

    switch (kind) {
        case Node_Any:       assert(ops.size() == 0); return to.any(type);
        case Node_Bottom:    assert(ops.size() == 0); return to.bottom(type);
        case Node_Enter:     assert(ops.size() == 1); return to.enter(    ops[0], name);
        case Node_Extract:   assert(ops.size() == 2); return to.extract(  ops[0], ops[1], name);
        case Node_Global:    assert(ops.size() == 1); return to.global(   ops[0], in->as<Global>()->is_mutable(), name);
        case Node_Hlt:       assert(ops.size() == 1); return to.hlt(      ops[0], name);
        case Node_Insert:    assert(ops.size() == 3); return to.insert(   ops[0], ops[1], ops[2], name);
        case Node_LEA:       assert(ops.size() == 2); return to.lea(      ops[0], ops[1], name);
        case Node_Leave:     assert(ops.size() == 2); return to.leave(    ops[0], ops[1], name);
        case Node_Load:      assert(ops.size() == 2); return to.load(     ops[0], ops[1], name);
        case Node_Map:       assert(ops.size() == 4); return to.map(      ops[0], ops[1], 
                                     in->as<Map>()->device(), in->as<Map>()->addr_space(), ops[2], ops[3], name);
        case Node_Unmap:     assert(ops.size() == 2); return to.unmap(    ops[0], ops[1], 
                                     in->as<Map>()->device(), in->as<Map>()->addr_space(),  name);
        case Node_Run:       assert(ops.size() == 1); return to.run(      ops[0], name);
        case Node_Select:    assert(ops.size() == 3); return to.select(   ops[0], ops[1], ops[2], name);
        case Node_Store:     assert(ops.size() == 3); return to.store(    ops[0], ops[1], ops[2], name);
        case Node_Tuple:                              return to.tuple(ops, name);
        case Node_Vector:                             return to.vector(ops, name);
        case Node_ArrayAgg:
            return to.array(type.as<ArrayType>()->elem_type(), ops, type.isa<DefiniteArrayType>(), name);
        case Node_Slot:    assert(ops.size() == 1);
            return to.slot(type.as<PtrType>()->referenced_type(), ops[0], in->as<Slot>()->index(), name);
        default: THORIN_UNREACHABLE;
    }
}

Type World::rebuild(World& to, Type type, ArrayRef<Type> elems) {
    if (elems.empty() && &type->world() == &to)
        return type;

    if (is_primtype(type->kind())) {
        assert(elems.size() == 0);
        auto primtype = type.as<PrimType>();
        return to.type(primtype->primtype_kind(), primtype->length());
    }

    switch (type->kind()) {
        case Node_DefiniteArrayType:
            assert(elems.size() == 0); 
            return to.definite_array_type(elems.front(), type.as<DefiniteArrayType>()->dim());
        case Node_TypeVar:              assert(elems.size() == 0); return to.type_var(type.as<TypeVar>()->index());
        case Node_IndefiniteArrayType:  assert(elems.size() == 1); return to.indefinite_array_type(elems.front());
        case Node_MemType:              assert(elems.size() == 0); return to.mem_type();
        case Node_FrameType:            assert(elems.size() == 0); return to.frame_type();
        case Node_PtrType: {
            assert(elems.size() == 1); 
            auto p = type.as<PtrType>();
            return to.ptr_type(elems.front(), p->length(), p->device(), p->addr_space());
        }
        case Node_TupleType:  return to.tuple_type(elems);
        case Node_FnType:     return to.fn_type(elems);
        default: THORIN_UNREACHABLE;
    }
}

const Param* World::param(Type type, Lambda* lambda, size_t index, const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    return new Param(gid_++, type, lambda, index, name);
}

/*
 * cse + unify
 */

const TypeNode* World::unify_base(const TypeNode* type) {
    auto i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        return *i;
    }

    auto p = types_.insert(type);
    assert(type->gid_ == size_t(-1));
    type->gid_ = gid_++;
    assert(p.second && "hash/equal broken");
    return type;
}

const DefNode* World::cse_base(const PrimOp* primop) {
    auto i = primops_.find(primop);
    if (i != primops_.end()) {
        for (size_t x = 0, e = primop->size(); x != e; ++x)
            primop->unregister_use(x);

        delete primop;
        primop = *i;
    } else {
        primop->set_gid(gid_++);
        auto p = primops_.insert(primop);
        assert(p.second && "hash/equal broken");
    }

    THORIN_CHECK_BREAK(primop->gid())
    return primop;
}

void World::destroy(Lambda* lambda) {
    assert(lambda->num_uses() == 0);
    assert(lambda->num_args() == 0);
    lambda->destroy_body();
    lambdas_.erase(lambda);
    delete lambda;
}

/*
 * optimizations
 */

void World::cleanup() { cleanup_world(*this); }

void World::opt() {
    cleanup();
    partial_evaluation(*this);
    merge_lambdas(*this);
    cleanup();
    lower2cff(*this);
    clone_bodies(*this);
    mem2reg(*this);
    lift_builtins(*this);
    memmap_builtins(*this);
    inliner(*this);
    merge_lambdas(*this);
    cleanup();
}

}
