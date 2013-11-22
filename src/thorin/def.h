#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"

namespace thorin {

//------------------------------------------------------------------------------

class DefNode;
class Lambda;
class PrimOp;
class Sigma;
class Tracker;
class Type;
class Use;
class World;

//------------------------------------------------------------------------------

/**
 * This class acts as a proxy for \p DefNode pointers.
 * This proxy hides that a \p DefNode may have been replaced by another one.
 * It automatically forwards to the replaced node.
 * If in doubt use a \p Def instead of \p DefNode*.
 * You almost never have to use a \p DefNode* directly.
 */
class Def {
public:
    Def() 
        : node_(nullptr)
    {}
    Def(const DefNode* node)
        : node_(node)
    {}

    bool empty() const { return node_ == nullptr; }
    const DefNode* node() const { return node_; }
    const DefNode* deref() const;
    const DefNode* operator *() const { return deref(); }
    bool operator == (const DefNode* other) const { return this->deref() == other; }
    bool operator == (const Use& use) const;
    bool operator == (Def other) const { return this->deref() == other.deref(); }
    bool operator != (Def other) const { return this->deref() != other.deref(); }
    operator const DefNode*() const { return deref(); }
    const DefNode* operator -> () const { return deref(); }

private:
    mutable const DefNode* node_;
};

/** 
 * References a user.
 * A \p Def u which uses \p Def d as i^th operand is a \p Use with \p index_ i of \p Def d.
 */
class Use {
public:
    Use() {}
    Use(size_t index, Def def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def& def() const { return def_; }
    bool operator == (Use use) const { return def() == use.def() && index() == use.index(); }
    bool operator != (Use use) const { return def() != use.def() || index() != use.index(); }
    bool operator < (Use) const;
    operator Def() const { return def_; }
    operator const DefNode*() const { return def_; }
    const Def& operator -> () const { return def_; }

private:
    size_t index_;
    Def def_;
};

inline bool Def::operator == (const Use& use) const { return this->deref() == use.def().deref(); }

struct UseHash { size_t operator () (Use use) const { return hash_combine(hash_value(use.def().node()), use.index()); } };
struct UseEqual { bool operator () (Use use1, Use use2) const { return use1 == use2; } };

class Peek {
public:
    Peek() {}
    Peek(Def def, Lambda* from)
        : def_(def)
        , from_(from)
    {}

    Def def() const { return def_; }
    Lambda* from() const { return from_; }

private:
    Def def_;
    Lambda* from_;
};

typedef std::vector<Peek> Peeks;

//------------------------------------------------------------------------------

/**
 * The base class for all three kinds of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Lambda%s.
 */
class DefNode : public MagicCast<DefNode> {
private:
    DefNode& operator = (const DefNode&); ///< Do not copy-assign a \p DefNode instance.
    DefNode(const DefNode&);              ///< Do not copy-construct a \p DefNode.

protected:
    DefNode(size_t gid, NodeKind kind, size_t size, const Type* type, bool is_const, const std::string& name)
        : kind_(kind)
        , ops_(size)
        , cur_pass_(0)
        , type_(type)
        , uses_(13) // 13 seems to perform best
        , representative_(this)
        , gid_(gid)
        , is_const_(is_const)
        , name(name)
    {}
    virtual ~DefNode() {}

    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;
    void resize(size_t n) { ops_.resize(n, nullptr); }

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    void set_op(size_t i, Def def);
    void unset_op(size_t i);
    void unset_ops();
    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;
    bool is_const() const { return is_const_; }
    /**
     * Returns the maximum depth of this \p Def%s depdency tree (induced by the \p ops).
     * \em const dependences are consideres leaves in this tree.
     * Thus, those dependences are not further propagted to determine the depth.
     */
    int non_const_depth() const;
    void dump() const;
    const PrimOp* is_non_const_primop() const;
    std::vector<Use> uses() const;
    bool is_proxy() const { return representative_ != this; }
    /// WARNING: slow!
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    const Type* type() const { return type_; }
    int order() const;
    bool is_generic() const;
    World& world() const;
    ArrayRef<Def> ops() const { return ops_; }
    Def op(size_t i) const { assert(i < ops().size()); return ops()[i]; }
    Def op_via_lit(Def def) const;
    void replace(Def) const;
    size_t length() const; ///< Returns the vector length. Raises an assertion if type of this is not a \p VectorType.
    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_minus_zero() const;
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }
    bool is_div()         const { return thorin::is_div  (kind()); }
    bool is_rem()         const { return thorin::is_rem  (kind()); }
    bool is_bitop()       const { return thorin::is_bitop(kind()); }
    bool is_shift()       const { return thorin::is_shift(kind()); }
    bool is_not()         const { return kind() == Node_xor && op(0)->is_allset(); }
    bool is_minus()       const { return (kind() == Node_sub || kind() == Node_fsub) && op(0)->is_minus_zero(); }
    bool is_div_or_rem()  const { return thorin::is_div_or_rem(kind()); }
    bool is_commutative() const { return thorin::is_commutative(kind()); }
    bool is_associative() const { return thorin::is_associative(kind()); }
    template<class T> inline T primlit_value() const; // implementation in literal.h

    // scratch operations

    size_t cur_pass() const { return cur_pass_; }
    bool visit(const size_t pass) const { 
        assert(cur_pass_ <= pass); 
        if (cur_pass_ != pass) { 
            cur_pass_ = pass; 
            return false; 
        } 
        return true; 
    }
    void visit_first(const size_t pass) const { assert(!is_visited(pass)); cur_pass_ = pass; }
    void unvisit(const size_t pass) const { assert(cur_pass_ == pass); --cur_pass_; }
    bool is_visited(const size_t pass) const { assert(cur_pass_ <= pass); return cur_pass_ == pass; }

private:
    NodeKind kind_;
    std::vector<Def> ops_;
    mutable size_t cur_pass_;
    // HACK
    mutable const Type* type_;
    mutable std::unordered_set<Use, UseHash, UseEqual> uses_;
    mutable const DefNode* representative_;
    mutable std::unordered_set<const DefNode*> representatives_of_;
    const size_t gid_;

protected:
    bool is_const_;

public:
    mutable std::string name; ///< Just do what ever you want with this field.

    /** 
     * Use this field in order to annotate information on this Def.
     * Various analyses have to memorize different stuff temporally.
     * Each analysis can use this field for its specific information. 
     * \attention { 
     *      Each pass/analysis simply overwrites this field again.
     *      So keep this in mind and perform copy operations in order to
     *      save your data before running the next pass/analysis.
     *      Also, keep in mind to perform clean up operations at the end 
     *      of your pass/analysis.
     * }
     */
    union {
        mutable void* ptr;
        mutable const void* cptr;
    };
    union {
        mutable bool flags[sizeof(size_t)/sizeof(bool)];
        mutable size_t counter;
    };

    friend class Def;
    friend class PrimOp;
    friend class World;
    friend void verify_closedness(World& world);
};

std::ostream& operator << (std::ostream& o, Def def);
inline bool Use::operator < (Use use) const { return def()->gid() < use.def()->gid() && index() < use.index(); }

//------------------------------------------------------------------------------

struct DefNodeHash { 
    size_t operator () (const DefNode* n) const { return hash_value(n->gid()); } 
};

struct DefNodeEqual { 
    bool operator () (const DefNode* n1, const DefNode* n2) const { return n1->gid() == n2->gid(); } 
};

//------------------------------------------------------------------------------

template<class Value>
class DefMap : public std::unordered_map<const DefNode*, Value, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_map<const DefNode*, Value, DefNodeHash, DefNodeEqual> Super;
};

template<class Value>
class DefMap<Value*> : public std::unordered_map<const DefNode*, Value*, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_map<const DefNode*, Value*, DefNodeHash, DefNodeEqual> Super;

    Value* find(const DefNode* def) const {
        auto i = Super::find(def);
        return i == Super::end() ? nullptr : i->second;
    }
};

class DefSet : public std::unordered_set<const DefNode*, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_set<const DefNode*, DefNodeHash, DefNodeEqual> Super;

    bool contains(const DefNode* def) { return Super::find(def) != Super::end(); }
    bool visit(const DefNode* def) { return !Super::insert(def).second; }
};

//------------------------------------------------------------------------------

typedef DefMap<const DefNode*> Def2Def;

class Param : public DefNode {
private:
    Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name)
        : DefNode(gid, Node_Param, 0, type, false, name)
        , lambda_(lambda)
        , index_(index)
    {}

public:
    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    Peeks peek() const;

private:
    mutable Lambda* lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

void mark_down(DefSet&, std::queue<Def>&);
void mark_down(const size_t pass, std::queue<Def>& queue);

//------------------------------------------------------------------------------

} // namespace thorin

#endif
