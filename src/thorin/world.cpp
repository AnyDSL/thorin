#include "thorin/world.h"

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
#include "thorin/transform/lift_enters.h"
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
    , tuple0_ (*unify(*join(new TupleTypeNode(*this, ArrayRef<Type>()))))
    , fn0_    (*unify(*join(new FnTypeNode   (*this, ArrayRef<Type>()))))
    , mem_    (*unify(*join(new MemTypeNode  (*this))))
    , frame_  (*unify(*join(new FrameTypeNode(*this))))
#define THORIN_ALL_TYPE(T, M) ,T##_(*unify(*join(new PrimTypeNode(*this, PrimType_##T, 1))))
#include "thorin/tables/primtypetable.h"
{}

World::~World() {
    for (auto primop : primops_) delete primop;
    for (auto lambda : lambdas_) delete lambda;
    for (auto type   : garbage_) delete type;
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
 * literals
 */

Def World::literal(PrimTypeKind kind, int64_t value, size_t length) {
    Def lit;
    switch (kind) {
#define THORIN_I_TYPE(T, M) case PrimType_##T:  lit = literal(T(value), 1); break;
#define THORIN_F_TYPE(T, M) THORIN_I_TYPE(T, M)
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

    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
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
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_sub:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_mul:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_div:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_rem:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() % r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_and:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_or:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_xor:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shl:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shr:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())));
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
            }
        } catch (BottomException) {
            return bottom(type);
        }
    }

    if (is_type_i(type)) {
        if (a == b) {
            switch (kind) {
                case ArithOp_add:
                    if (is_type_i(type))
                        return arithop_mul(cond, literal(type, 2), a);
                    else
                        break;

                case ArithOp_sub:
                case ArithOp_xor: return zero(type);

                case ArithOp_and:
                case ArithOp_or:  return a;

                case ArithOp_div:
                    if (b->is_zero())
                        return bottom(type);
                    return one(type);

                case ArithOp_rem:
                    if (b->is_zero())
                        return bottom(type);
                    return zero(type);

                default: break;
            }
        }

        if (b->is_zero()) {
            switch (kind) {
                case ArithOp_div:
                case ArithOp_rem: return bottom(type);

                case ArithOp_shl:
                case ArithOp_shr: return a;

                default: break;
            }
        } else if (b->is_one()) {
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

    if (is_type_i(kind)) {
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
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(kind)) { // TODO properly obey floats
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
#define THORIN_F_TYPE(T, M) \
        case PrimType_##T: \
            return arithop_sub(cond, literal_##T(-0.f, def->length()), def);
#include "thorin/tables/primtypetable.h"
        default:
            assert(is_type_i(kind));
            return arithop_sub(cond, zero(kind), def);
    }
}

Def World::cmp(CmpKind kind, Def cond, Def a, Def b, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
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
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() == r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_ne:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() != r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_lt:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <  r.get_##T());
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_le:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <= r.get_##T());
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

Def World::cast(Def cond, Def from, Type to, const std::string& name) {
    if (cond->isa<Bottom>() || from->isa<Bottom>())
        return bottom(to);

    if (auto vec = from->isa<Vector>()) {
        size_t num = vec->length();
        auto to_vec = to.as<VectorType>();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cast(vec->op(i), to_vec->scalarize());
        return vector(ops, name);
    }

    auto lit = from->isa<PrimLit>();
    auto to_type = to.isa<PrimType>();
    if (lit && to_type) {
        Box box = lit->value();

        switch (lit->primtype_kind()) {
            case PrimType_bool:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_bool()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps8:
            case PrimType_qs8:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_s8()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps16:
            case PrimType_qs16:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_s16()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps32:
            case PrimType_qs32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_s32()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps64:
            case PrimType_qs64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_s64()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu8:
            case PrimType_qu8:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_u8()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu16:
            case PrimType_qu16:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_u16()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu32:
            case PrimType_qu32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_u32()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu64:
            case PrimType_qu64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_u64()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pf32:
            case PrimType_qf32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_f32()));
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pf64:
            case PrimType_qf64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(T(box.get_f64()));
#include "thorin/tables/primtypetable.h"
                }
        }
    }

    return cse(new Cast(cond, from, to, name));
}

Def World::bitcast(Def cond, Def from, Type to, const std::string& name) {
    if (cond->isa<Bottom>() || from->isa<Bottom>())
        return bottom(to);

    if (auto vec = from->isa<Vector>()) {
        size_t num = vec->length();
        auto to_vec = to.as<VectorType>();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = bitcast(vec->op(i), to_vec->scalarize());
        return vector(ops, name);
    }

    // TODO constant folding
    return cse(new Bitcast(cond, from, to , name));
}

Def World::extract(Def agg, Def index, const std::string& name) {
    if (agg->isa<Bottom>())
        return bottom(Extract::type(agg, index));

    if (auto load = agg->isa<Load>())
        return this->load(load->mem(), lea(load->ptr(), index, load->name), name);

    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto lit = index->isa<PrimLit>()) {
            if (!agg->isa<IndefiniteArray>())
                return aggregate->op(lit);
        }
    }

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
            if (!agg->isa<IndefiniteArray>()) {
                Array<Def> args(agg->size());
                std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
                args[literal->primlit_value<u64>()] = value;
                return rebuild(aggregate, args);
            }
        }
    }

    return cse(new Insert(agg, index, value, name));
}

Def World::alloc(Def mem, Type type, Def extra, const std::string& name) { return cse(new Alloc(mem, type, extra, name)); }
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

const Map* World::map(Def mem, Def ptr, uint32_t device, AddressSpace addr_space, Def mem_offset, Def mem_size, const std::string& name) {
    return cse(new Map(mem, ptr, device, addr_space, mem_offset, mem_size, name));
}

const Map* World::map(Def mem, Def ptr, Def device, Def addr_space, Def mem_offset, Def mem_size, const std::string& name) {
    return map(mem, ptr, device->as<PrimLit>()->ps32_value().data(),
            (AddressSpace)addr_space->as<PrimLit>()->ps32_value().data(), mem_offset, mem_size, name);
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

    if (auto insert = value->isa<Insert>())
        return store(mem, lea(ptr, insert->index(), insert->name), value, name);

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

    return global(definite_array(str_array), false, name);
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

Def World::end_run(Def def, Def run, const std::string& name) { return cse(new EndRun(def, run, name)); }
Def World::end_hlt(Def def, Def hlt, const std::string& name) { return cse(new EndHlt(def, hlt, name)); }

Lambda* World::lambda(FnType fn, Lambda::Attribute attribute, Lambda::Attribute intrinsic, const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto l = new Lambda(gid_++, fn, attribute, intrinsic, true, name);
    lambdas_.insert(l);

    size_t i = 0;
    for (auto arg : fn->args())
        l->params_.push_back(param(arg, l, i++));

    return l;
}

Lambda* World::meta_lambda() {
    auto l = lambda(fn_type(), "meta");
    l->jump(bottom(fn_type()), {});
    return l;
}

Lambda* World::basicblock(const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto bb = new Lambda(gid_++, fn_type(), Lambda::Attribute(0), Lambda::Attribute(0), false, name);
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
    if (is_primtype(kind)) {
        assert(ops.size() == 0);
        auto primlit = in->as<PrimLit>();
        return to.literal(primlit->primtype_kind(), primlit->value());
    }

    switch (kind) {
        case Node_Alloc:    assert(ops.size() == 2); return to.alloc(   ops[0], type.as<PtrType>()->referenced_type(), ops[1], name);
        case Node_Any:      assert(ops.size() == 0); return to.any(type);
        case Node_Bottom:   assert(ops.size() == 0); return to.bottom(type);
        case Node_Bitcast:  assert(ops.size() == 2); return to.bitcast( ops[0], ops[1], type);
        case Node_Cast:     assert(ops.size() == 2); return to.cast(    ops[0], ops[1], type);
        case Node_Enter:    assert(ops.size() == 1); return to.enter(   ops[0], name);
        case Node_Extract:  assert(ops.size() == 2); return to.extract( ops[0], ops[1], name);
        case Node_Global:   assert(ops.size() == 1); return to.global(  ops[0], in->as<Global>()->is_mutable(), name);
        case Node_Hlt:      assert(ops.size() == 1); return to.hlt(     ops[0], name);
        case Node_EndHlt:   assert(ops.size() == 2); return to.end_hlt( ops[0], ops[1], name);
        case Node_EndRun:   assert(ops.size() == 2); return to.end_run( ops[0], ops[1], name);
        case Node_Insert:   assert(ops.size() == 3); return to.insert(  ops[0], ops[1], ops[2], name);
        case Node_LEA:      assert(ops.size() == 2); return to.lea(     ops[0], ops[1], name);
        case Node_Leave:    assert(ops.size() == 2); return to.leave(   ops[0], ops[1], name);
        case Node_Load:     assert(ops.size() == 2); return to.load(    ops[0], ops[1], name);
        case Node_Map:      assert(ops.size() == 4); return to.map(     ops[0], ops[1],
                                    in->as<Map>()->device(), in->as<Map>()->addr_space(), ops[2], ops[3], name);
        case Node_Unmap:    assert(ops.size() == 2); return to.unmap(   ops[0], ops[1],
                                    in->as<Map>()->device(), in->as<Map>()->addr_space(),  name);
        case Node_Run:      assert(ops.size() == 1); return to.run(     ops[0], name);
        case Node_Select:   assert(ops.size() == 3); return to.select(  ops[0], ops[1], ops[2], name);
        case Node_Store:    assert(ops.size() == 3); return to.store(   ops[0], ops[1], ops[2], name);
        case Node_StructAgg:                         return to.struct_agg(type.as<StructAppType>(), ops, name);
        case Node_Tuple:                             return to.tuple(ops, name);
        case Node_Vector:                            return to.vector(ops, name);
        case Node_DefiniteArray:
            return to.definite_array(type.as<DefiniteArrayType>()->elem_type(), ops, name);
        case Node_IndefiniteArray: assert(ops.size() == 1);
            return to.indefinite_array(type.as<IndefiniteArrayType>()->elem_type(), ops[0], name);
        case Node_Slot:    assert(ops.size() == 1);
            return to.slot(type.as<PtrType>()->referenced_type(), ops[0], in->as<Slot>()->index(), name);
        default: THORIN_UNREACHABLE;
    }
}

Type World::rebuild(World& to, Type type, ArrayRef<Type> args) {
    if (args.empty() && &type->world() == &to)
        return type;

    if (is_primtype(type->kind())) {
        assert(args.size() == 0);
        auto primtype = type.as<PrimType>();
        return to.type(primtype->primtype_kind(), primtype->length());
    }

    switch (type->kind()) {
        case Node_DefiniteArrayType: {
            assert(args.size() == 1);
            return to.definite_array_type(args.front(), type.as<DefiniteArrayType>()->dim());
        }
        case Node_TypeVar:              assert(args.size() == 0); return to.type_var();
        case Node_IndefiniteArrayType:  assert(args.size() == 1); return to.indefinite_array_type(args.front());
        case Node_MemType:              assert(args.size() == 0); return to.mem_type();
        case Node_FrameType:            assert(args.size() == 0); return to.frame_type();
        case Node_PtrType: {
            assert(args.size() == 1);
            auto p = type.as<PtrType>();
            return to.ptr_type(args.front(), p->length(), p->device(), p->addr_space());
        }
        case Node_StructAbsType: {
            // TODO how do we handle recursive types?
            auto ntype = to.struct_abs_type(args.size());
            for (size_t i = 0, e = args.size(); i != e; ++i)
                ntype->set(i, args[i]);
            return ntype;
        }
        case Node_StructAppType: {
            assert(args.size() >= 1);
            return to.struct_app_type(args[0].as<StructAbsType>(), args.slice_from_begin(1));
        }
        case Node_TupleType:        return to.tuple_type(args);
        case Node_FnType:           return to.fn_type(args);
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
    assert(type->is_closed());
    if (type->is_unified())
        return type->representative();

    auto i = types_.find(type);
    if (i != types_.end()) {
        auto representative = *i;
        type->representative_ = representative;
        return representative;
    } else {
        auto p = types_.insert(type);
        assert(p.second && "hash/equal broken");
        type->representative_ = type;
        return type;
    }
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
    memmap_builtins(*this);
    lift_builtins(*this);
    inliner(*this);
    merge_lambdas(*this);
    lift_enters(*this);
    cleanup();
}

}
