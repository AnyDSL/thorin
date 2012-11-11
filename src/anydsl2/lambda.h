#ifndef ANYDSL2_LAMBDA_H
#define ANYDSL2_LAMBDA_H

#include <set>
#include <vector>
#include <functional>

#include "anydsl2/def.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class GenericMap;
class Lambda;
class Pi;

struct LambdaLT : public std::binary_function<Lambda*, Lambda*, bool> {
    inline bool operator () (Lambda* l1, Lambda* l2) const;
};

typedef std::set<Lambda*, LambdaLT> LambdaSet;

typedef std::vector<const Param*> Params;

class Lambda : public Def {
private:

    Lambda(size_t gid, const Pi* pi, uint32_t flags);
    virtual ~Lambda();

public:

    enum Flags {
        Extern = 1 << 0,
    };

    Lambda* stub(const GenericMap& generic_map) const;

    const Param* append_param(const Type* type);

    Array<const Param*> fo_params() const;
    Array<const Param*> ho_params() const;
    Array<const Def*> fo_args() const;
    Array<const Def*> ho_args() const;
    bool is_returning() const;
    bool is_bb() const;

    LambdaSet targets() const;
    LambdaSet hos() const;
    LambdaSet succs() const;
    LambdaSet preds() const;
    LambdaSet direct_preds() const;
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { return params_[i]; }
    const Def* to() const { return op(0); };
    Defs args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    const Pi* arg_pi() const;
    uint32_t flags() const { return flags_; }
    size_t gid() const { return gid_; }
    size_t sid() const { return sid_; }
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }

    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;

    void dump(bool fancy = false, int indent = 0) const;

    bool is_extern() const { return flags_ & Extern; }
    void set_extern() { flags_ |= Extern; }

    bool sid_valid() { return sid_ != size_t(-1); }
    bool sid_invalid() { return sid_ == size_t(-1); }
    void invalidate_sid() { sid_ = size_t(-1); }

    void jump(const Def* to, ArrayRef<const Def*> args);
    void jump0(const Def* to) {
        return jump(to, ArrayRef<const Def*>(0, 0));
    }
    void jump1(const Def* to, const Def* arg1) {
        const Def* args[1] = { arg1 };
        return jump(to, args);
    }
    void jump2(const Def* to, const Def* arg1, const Def* arg2) {
        const Def* args[2] = { arg1, arg2 };
        return jump(to, args);
    }
    void jump3(const Def* to, const Def* arg1, const Def* arg2, const Def* arg3) {
        const Def* args[3] = { arg1, arg2, arg3 };
        return jump(to, args);
    }
    void branch(const Def* cond, const Def* tto, const Def* fto);

    Lambda* drop(ArrayRef<size_t> indices, ArrayRef<const Def*> with, bool self);
    Lambda* drop(ArrayRef<size_t> indices, ArrayRef<const Def*> with, const GenericMap& generic_map, bool self);

private:

    template<bool fo> Array<const Param*> classify_params() const;
    template<bool fo> Array<const Def*> classify_args() const;

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    size_t gid_; ///< global index
    uint32_t flags_;
    Params params_;
    size_t sid_; ///< scope index

    friend class World;
    friend class Param;
    friend class Scope;
};

bool LambdaLT::operator () (Lambda* l1, Lambda* l2) const { return l1->gid() < l2->gid(); };

} // namespace anydsl2

#endif
