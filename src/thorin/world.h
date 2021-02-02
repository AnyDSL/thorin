#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/enums.h"
#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"
#include "thorin/config.h"

namespace thorin {

enum class LogLevel { Debug, Verbose, Info, Warn, Error };

/**
 * The World represents the whole program and manages creation and destruction of Thorin nodes.
 * In particular, the following things are done by this class:
 *
 *  - @p Type unification: \n
 *      There exists only one unique @p Type.
 *      These @p Type%s are hashed into an internal map for fast access.
 *      The getters just calculate a hash and lookup the @p Type, if it is already present, or create a new one otherwise.
 *  - Value unification: \n
 *      This is a built-in mechanism for the following things:
 *      - constant pooling
 *      - constant folding
 *      - common subexpression elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *
 *  @p PrimOp%s do not explicitly belong to a Continuation.
 *  Instead they either implicitly belong to a Continuation--when
 *  they (possibly via multiple levels of indirection) depend on a Continuation's Param--or they are dead.
 *  Use @p cleanup to remove dead code and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *  This is particular useful for multi-threading.
 */
class World : public TypeTable, public Streamable<World> {
public:
    typedef HashSet<const PrimOp*, PrimOpHash> PrimOpSet;

    struct BreakHash {
        static uint64_t hash(size_t i) { return i; }
        static bool eq(size_t i1, size_t i2) { return i1 == i2; }
        static size_t sentinel() { return size_t(-1); }
    };

    typedef HashSet<size_t, BreakHash> Breakpoints;

    World(World&&) = delete;
    World& operator=(const World&) = delete;

    explicit World(const std::string& name = {});
    ///  Inherits the @p state_ of the @p other @p World but does @em not perform a copy.
    explicit World(const World& other)
        : World(other.name())
    {
        stream_ = other.stream_;
        state_  = other.state_;
    }
    ~World();

    // literals

#define THORIN_ALL_TYPE(T, M) \
    const Def* literal_##T(T val, Debug dbg, size_t length = 1) { return literal(PrimType_##T, Box(val), dbg, length); }
#include "thorin/tables/primtypetable.h"
    const Def* literal(PrimTypeTag tag, Box box, Debug dbg, size_t length = 1) { return splat(cse(new PrimLit(*this, tag, box, dbg)), length); }
    template<class T>
    const Def* literal(T value, Debug dbg = {}, size_t length = 1) { return literal(type2tag<T>::tag, Box(value), dbg, length); }
    const Def* zero(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 0, dbg, length); }
    const Def* zero(const Type* type, Debug dbg = {}, size_t length = 1) { return zero(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* one(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 1, dbg, length); }
    const Def* one(const Type* type, Debug dbg = {}, size_t length = 1) { return one(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* allset(PrimTypeTag tag, Debug dbg = {}, size_t length = 1);
    const Def* allset(const Type* type, Debug dbg = {}, size_t length = 1) { return allset(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* top(const Type* type, Debug dbg = {}, size_t length = 1) { return splat(cse(new Top(type, dbg)), length); }
    const Def* bottom(const Type* type, Debug dbg = {}, size_t length = 1) { return splat(cse(new Bottom(type, dbg)), length); }
    const Def* bottom(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return bottom(prim_type(tag), dbg, length); }

    // arithops

    /// Creates an \p ArithOp or a \p Cmp.
    const Def* binop(int tag, const Def* lhs, const Def* rhs, Debug dbg = {});
    const Def* arithop_not(const Def* def, Debug dbg = {});
    const Def* arithop_minus(const Def* def, Debug dbg = {});
    const Def* arithop(ArithOpTag tag, const Def* lhs, const Def* rhs, Debug dbg = {});
#define THORIN_ARITHOP(OP) \
    const Def* arithop_##OP(const Def* lhs, const Def* rhs, Debug dbg = {}) { \
        return arithop(ArithOp_##OP, lhs, rhs, dbg); \
    }
#include "thorin/tables/arithoptable.h"

    // compares

    const Def* cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg = {});
#define THORIN_CMP(OP) \
    const Def* cmp_##OP(const Def* lhs, const Def* rhs, Debug dbg = {}) { \
        return cmp(Cmp_##OP, lhs, rhs, dbg);  \
    }
#include "thorin/tables/cmptable.h"

    // casts

    const Def* convert(const Type* to, const Def* from, Debug dbg = {});
    const Def* cast(const Type* to, const Def* from, Debug dbg = {});
    const Def* bitcast(const Type* to, const Def* from, Debug dbg = {});

    // aggregate operations

    const Def* definite_array(const Type* elem, Defs args, Debug dbg = {}) {
        return try_fold_aggregate(cse(new DefiniteArray(*this, elem, args, dbg)));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs args, Debug dbg = {}) {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, dbg);
    }
    const Def* indefinite_array(const Type* elem, const Def* dim, Debug dbg = {}) {
        return cse(new IndefiniteArray(*this, elem, dim, dbg));
    }
    const Def* struct_agg(const StructType* struct_type, Defs args, Debug dbg = {}) {
        return try_fold_aggregate(cse(new StructAgg(struct_type, args, dbg)));
    }
    const Def* tuple(Defs args, Debug dbg = {}) { return args.size() == 1 ? args.front() : try_fold_aggregate(cse(new Tuple(*this, args, dbg))); }

    const Def* variant(const VariantType* variant_type, const Def* value, size_t index, Debug dbg = {}) { return cse(new Variant(variant_type, value, index, dbg)); }
    const Def* variant_index  (const Def* value, Debug dbg = {});
    const Def* variant_extract(const Def* value, size_t index, Debug dbg = {});

    const Def* closure(const ClosureType* closure_type, const Def* fn, const Def* env, Debug dbg = {}) { return cse(new Closure(closure_type, fn, env, dbg)); }
    const Def* vector(Defs args, Debug dbg = {}) {
        if (args.size() == 1) return args[0];
        return try_fold_aggregate(cse(new Vector(*this, args, dbg)));
    }
    /// Splats \p arg to create a \p Vector with \p length.
    const Def* splat(const Def* arg, size_t length = 1, Debug dbg = {});
    const Def* extract(const Def* tuple, const Def* index, Debug dbg = {});
    const Def* extract(const Def* tuple, u32 index, Debug dbg = {}) {
        return extract(tuple, literal_qu32(index, dbg), dbg);
    }
    const Def* insert(const Def* tuple, const Def* index, const Def* value, Debug dbg = {});
    const Def* insert(const Def* tuple, u32 index, const Def* value, Debug dbg = {}) {
        return insert(tuple, literal_qu32(index, dbg), value, dbg);
    }

    const Def* select(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    const Def* align_of(const Type* type, Debug dbg = {});
    const Def* size_of(const Type* type, Debug dbg = {});

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Type* type, const Def* frame, Debug dbg = {}) { return cse(new Slot(type, frame, dbg)); }
    const Def* alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg = {});
    const Def* alloc(const Type* type, const Def* mem, Debug dbg = {}) { return alloc(type, mem, literal_qu64(0, dbg), dbg); }
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    const Def* lea(const Def* ptr, const Def* index, Debug dbg);
    const Assembly* assembly(const Type* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    const Assembly* assembly(Types types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});

    // partial evaluation related stuff

    const Def* hlt(const Def* def, Debug dbg = {});
    const Def* known(const Def* def, Debug dbg = {});
    const Def* run(const Def* def, Debug dbg = {});

    // continuations

    Continuation* continuation(const FnType*, Continuation::Attributes, Debug = {});
    Continuation* continuation(const FnType* fn_type, Debug dbg = {}) {
        return continuation(fn_type, Continuation::Attributes(), dbg);
    }
    Continuation* continuation(Debug dbg = {}) { return continuation(fn_type(), dbg); }
    Continuation* branch() const { return branch_; }
    Continuation* match(const Type* type, size_t num_patterns);
    Continuation* end_scope() const { return end_scope_; }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // getters

    const std::string& name() const { return name_; }
    const PrimOpSet& primops() const { return primops_; }
    const ContinuationSet& continuations() const { return continuations_; }
    Array<Continuation*> copy_continuations() const;
    Array<Continuation*> exported_continuations() const;
    bool empty() const { return continuations().size() <= 2; } // TODO rework intrinsic stuff. 2 = branch + end_scope

    /// @name partial evaluation done?
    //@{
    void mark_pe_done(bool flag = true) { state_.pe_done = flag; }
    bool is_pe_done() const { return state_.pe_done; }
    //@}

#if THORIN_ENABLE_CHECKS
    /// @name debugging features
    //@{
    void     breakpoint(size_t number);
    void use_breakpoint(size_t number);
    void enable_history(bool flag = true);
    bool track_history() const;
    const Def* gid2def(u32 gid);
    //@}
#endif

    /// @name logging
    //@{
    Stream& stream(Stream&) const;
    Stream& stream() { return *stream_; }
    LogLevel min_level() const { return state_.min_level; }

    void set(LogLevel min_level) { state_.min_level = min_level; }
    void set(std::shared_ptr<Stream> stream) { stream_ = stream; }

    template<class... Args>
    void log(LogLevel level, Loc loc, const char* fmt, Args&&... args) {
        if (stream_ && int(min_level()) <= int(level)) {
            stream().fmt("{}:{}: ", colorize(level2string(level), level2color(level)), colorize(loc.to_string(), 7));
            stream().fmt(fmt, std::forward<Args&&>(args)...).endl().flush();
        }
    }

    template<class... Args>
    [[noreturn]] void error(Loc loc, const char* fmt, Args&&... args) {
        log(LogLevel::Error, loc, fmt, std::forward<Args&&>(args)...);
        std::abort();
    }

    template<class... Args> void idef(const Def* def, const char* fmt, Args&&... args) { log(LogLevel::Info, def->debug().loc, fmt, std::forward<Args&&>(args)...); }
    template<class... Args> void wdef(const Def* def, const char* fmt, Args&&... args) { log(LogLevel::Warn, def->debug().loc, fmt, std::forward<Args&&>(args)...); }
    template<class... Args> void edef(const Def* def, const char* fmt, Args&&... args) { error(def->debug().loc, fmt, std::forward<Args&&>(args)...); }

    static const char* level2string(LogLevel level);
    static int level2color(LogLevel level);
    static std::string colorize(const std::string& str, int color);
    //@}

    friend void swap(World& w1, World& w2) {
        using std::swap;
        swap(static_cast<TypeTable&>(w1), static_cast<TypeTable&>(w2));
        swap(w1.name_,          w2.name_);
        swap(w1.continuations_, w2.continuations_);
        swap(w1.primops_,       w2.primops_);
        swap(w1.branch_,        w2.branch_);
        swap(w1.end_scope_,     w2.end_scope_);
        swap(w1.state_,         w2.state_);
    }

private:
    const Param* param(const Type* type, Continuation* continuation, size_t index, Debug dbg);
    const Def* try_fold_aggregate(const Aggregate*);
    const Def* cse_base(const PrimOp*);
    template<class T> const T* cse(const T* primop) { return cse_base(primop)->template as<T>(); }

    struct State {
        LogLevel min_level = LogLevel::Error;
        u32 cur_gid = 0;
        bool pe_done = false;
#if THORIN_ENABLE_CHECKS
        bool track_history = false;
        Breakpoints breakpoints;
        Breakpoints use_breakpoints;
#endif
    } state_;

    std::string name_;
    ContinuationSet continuations_;
    PrimOpSet primops_;
    Continuation* branch_;
    Continuation* end_scope_;
    std::shared_ptr<Stream> stream_;

    friend class Cleaner;
    friend class Continuation;
    friend void Def::replace(Tracker) const;
};

}

#define ELOG(...) log(thorin::LogLevel::Error,   thorin::Loc(__FILE__, {__LINE__, thorin::u32(-1)}, {__LINE__, thorin::u32(-1)}), __VA_ARGS__)
#define WLOG(...) log(thorin::LogLevel::Warn,    thorin::Loc(__FILE__, {__LINE__, thorin::u32(-1)}, {__LINE__, thorin::u32(-1)}), __VA_ARGS__)
#define ILOG(...) log(thorin::LogLevel::Info,    thorin::Loc(__FILE__, {__LINE__, thorin::u32(-1)}, {__LINE__, thorin::u32(-1)}), __VA_ARGS__)
#define VLOG(...) log(thorin::LogLevel::Verbose, thorin::Loc(__FILE__, {__LINE__, thorin::u32(-1)}, {__LINE__, thorin::u32(-1)}), __VA_ARGS__)
#ifndef NDEBUG
#define DLOG(...) log(thorin::LogLevel::Debug,   thorin::Loc(__FILE__, {__LINE__, thorin::u32(-1)}, {__LINE__, thorin::u32(-1)}), __VA_ARGS__)
#else
#define DLOG(...) do {} while (false)
#endif

#endif
